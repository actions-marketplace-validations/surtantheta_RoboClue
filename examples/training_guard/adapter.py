"""Passive provenance and bounded cleanup around the real GR00T iterator."""
import hashlib
import json
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import MethodType


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


@dataclass
class SampleRecord:
    original: object
    source_key: str
    provenance: dict


class Provenance:
    """Observe original sample identities without inserting or changing fields."""

    def __init__(self,dataset,context):
        self.dataset,self.context=dataset,context
        self.local=threading.local()
        self.lock=threading.Lock()
        self.samples={}
        self.shards=[]
        self.capture_enabled=True
        self.original_shard=dataset.get_shard
        self.original_datapoint=dataset.get_datapoint
        owner=self

        def shard(instance,index):
            with owner.lock:
                capture=owner.capture_enabled
            if not capture:
                owner.local.passthrough=True
                try:
                    return owner.original_shard(index)
                finally:
                    owner.local.passthrough=False
            expected=deque((int(position),int(step)) for position,steps in instance.sharded_episodes[index]
                           for step in steps)
            assert len(expected)==1,'Pilot metadata bound requires exactlyone native sample per shard'
            assert not getattr(owner.local,'active',False)
            owner.local.active=True
            owner.local.expected=expected
            owner.local.returned=[]
            try:
                original=owner.original_shard(index)
                assert not expected
                assert len(original)==len(owner.local.returned)
                assert all(left is right for left,right in zip(original,owner.local.returned,strict=True))
                with owner.lock:
                    owner.shards.append({'shard':int(index),'original_list_identity':id(original),
                        'sample_keys':[owner.samples[id(item)].source_key for item in original]})
                return original
            finally:
                owner.local.active=False

        def datapoint(instance,episode_data,step_index):
            if getattr(owner.local,'passthrough',False):
                return owner.original_datapoint(episode_data,step_index)
            assert getattr(owner.local,'active',False) and owner.local.expected
            position,step=owner.local.expected.popleft()
            assert int(step_index)==step
            original=owner.original_datapoint(episode_data,step_index)
            provenance=owner.describe(position,step,len(episode_data))
            key='sha256:'+digest(provenance['source'])
            with owner.lock:
                assert id(original) not in owner.samples
                owner.samples[id(original)]=SampleRecord(original,key,provenance)
            owner.local.returned.append(original)
            return original

        dataset.get_shard=MethodType(shard,dataset)
        dataset.get_datapoint=MethodType(datapoint,dataset)

    def describe(self,position,step,length):
        import pyarrow.parquet as pq

        loader=self.dataset.episode_loader
        meta=loader.episodes_metadata[position]
        stored=int(meta['episode_index'])
        assert length==meta['length']
        relative=loader.data_path_pattern.format(episode_chunk=stored//loader.chunk_size,episode_index=stored)
        parquet=Path(loader.dataset_path)/relative
        assert parquet.resolve().is_relative_to(Path(loader.dataset_path).resolve())
        assert pq.ParquetFile(parquet).metadata.num_rows==length,'Native nominal truncation is not source coverage'
        fields=[]
        rows={}
        for role in ('state','action'):
            config=self.dataset.modality_configs[role]
            selected=[step+int(delta) for delta in config.delta_indices]
            assert all(0<=index<length for index in selected)
            rows[role]=selected
            offset=0
            for index,key in enumerate(config.modality_keys):
                declaration=loader.modality_meta[role][key]
                begin,end=int(declaration['start']),int(declaration['end'])
                field={'role':role,'key':key,'column':declaration.get('original_key') or
                    {'state':'observation.state','action':'action'}[role],
                    'source_slice':[begin,end],'packed_slice':[offset,offset+end-begin]}
                if role=='action':
                    action=config.action_configs[index]
                    field.update(declared_representation=action.rep.value,
                                 reference_state_key=action.state_key or key)
                fields.append(field)
                offset+=end-begin
        video=[]
        for key in self.dataset.modality_configs['video'].modality_keys:
            meta_key=loader._video_key_mapping.get(key,key)
            original=loader.modality_meta['video'][meta_key].get('original_key','observation.images.'+meta_key)
            name=loader.video_path_pattern.format(episode_chunk=stored//loader.chunk_size,
                                                episode_index=stored,video_key=original)
            path=Path(loader.dataset_path)/name
            video.append({'key':key,'relative_path':name,'sha256':self.context['files'][str(path)],
                'physical_rows':[step+int(delta) for delta in self.dataset.modality_configs['video'].delta_indices]})
        source={'dataset_content_sha256':self.context['dataset_content_sha256'],
            'embodiment':self.dataset.embodiment_tag.value,'metadata_position':position,'stored_episode_index':stored,
            'physical_step':step,'rows':rows,'columns':fields,'parquet_path':relative,
            'parquet_sha256':self.context['files'][str(parquet)],'video':video,'allow_padding':False}
        return {'source':source,'processor':self.context['processor'],
            'native_source':self.context['native_source'],'scope':'Declared native source sample; no historical or physical authentication'}

    def records(self,features):
        with self.lock:
            values=[]
            for original in features:
                record=self.samples.get(id(original))
                assert record is not None and record.original is original,'Unobserved or substituted native sample'
                values.append(record)
            return values

    def source_keys(self,features):
        return [record.source_key for record in self.records(features)]

    def stop_capture(self):
        # Finish any already-started native shard coherently; future shards
        # pass through without retaining source samples after the startup gate.
        with self.lock:
            self.capture_enabled=False

    def restore(self):
        self.dataset.get_shard=self.original_shard
        self.dataset.get_datapoint=self.original_datapoint


class CollatorTap:
    """Retain the exact native result for receipt/identity checks, without rewriting it."""

    def __init__(self,native,provenance):
        self.native,self.provenance=native,provenance
        self.last_batch=None
        self.last_records=None
        self.calls=0

    def __call__(self,features):
        self.last_records=self.provenance.records(features)
        original=self.native(features)
        self.last_batch=original
        self.calls+=1
        return original


class NativeSession:
    def __init__(self,dataset,mixture,native_collator,context,contract=None,*,max_samples=2,loader=None):
        import torch
        from torch.utils.data import DataLoader

        from policy_doctor.training_guard import GuardedCollator

        assert type(max_samples) is int and 1<=max_samples<=3
        if loader is not None:
            if (not isinstance(loader,DataLoader) or loader.dataset is not mixture
                    or loader.num_workers!=0 or loader.batch_size!=1 or loader.drop_last
                    or loader.collate_fn is not native_collator):
                raise ValueError('Startup probe requires the original native DataLoader, batch_size=1, num_workers=0, drop_last=False')
        self.max_samples=max_samples
        self.dataset,self.mixture=dataset,mixture
        self.provenance=Provenance(dataset,context)
        self.tap=CollatorTap(native_collator,self.provenance)
        self.guard=GuardedCollator(self.tap,contract,source_keys=self.provenance.source_keys) if contract else None
        self.original_collator=native_collator
        self.existing_loader=loader is not None
        self.loader=loader if loader is not None else DataLoader(mixture,batch_size=1,num_workers=0,
            collate_fn=native_collator,generator=torch.Generator().manual_seed(42))
        self.loader.collate_fn=self.guard or self.tap
        self.iterator=iter(self.loader)
        self.cleanup=None

    def next(self):
        original=next(self.iterator)
        assert original is self.tap.last_batch,'Guard must return the exact native batch object'
        return original

    def close(self):
        errors=[]
        future=getattr(self.mixture,'_cache_job',None)
        executor=getattr(self.mixture,'_executor',None)
        cancelled=False
        try:
            generator=getattr(getattr(self.iterator,'_dataset_fetcher',None),'dataset_iter',None)
            if generator is not None:
                generator.close()
            if future is not None:
                cancelled=future.cancel()
                if not cancelled:
                    try:
                        future.result(timeout=60)
                    except BaseException as error:
                        errors.append({'type':type(error).__name__,'message':str(error)})
        finally:
            if executor is not None:
                threads=list(executor._threads)
                executor.shutdown(wait=True,cancel_futures=True)
                assert all(not thread.is_alive() for thread in threads)
            self.provenance.restore()
            self.loader.collate_fn=self.original_collator
        self.cleanup={'pending_cancelled':cancelled,'background_errors':errors,
            'produced_shards':len(self.provenance.shards),'produced_samples':len(self.provenance.samples),
            'native_batches_collated':self.tap.calls,'requested_sample_limit':self.max_samples,
            'maximum_produced_samples':self.max_samples+1,
            'prefetched_unassessed_samples':len(self.provenance.samples)-self.tap.calls,
            'shards':self.provenance.shards,'executor_shutdown':True}
        assert self.tap.calls<=self.max_samples
        assert len(self.provenance.shards)<=self.max_samples+1 and len(self.provenance.samples)<=self.max_samples+1
        if errors:
            raise RuntimeError('Native prefetch failed: '+json.dumps(errors))
        return self.cleanup
