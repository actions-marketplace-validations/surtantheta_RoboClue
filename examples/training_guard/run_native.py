"""Real GR00T mixture/collator readiness and a pinned target-preservation loop.

Run only under the separately reviewed CPU manifest. No model or weights run.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import inspect
import json
import os
import random
import sys
import traceback
from collections.abc import Mapping
from pathlib import Path


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):
            result.update(block)
    return result.hexdigest()


def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def load(path):
    return json.loads(Path(path).read_text())


def newdir(path):
    path=Path(path)
    assert not any(item.is_symlink() for item in (path,*path.parents))
    path.mkdir(mode=0o700)


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')


def tree(root,links):
    root=Path(root)
    assert root.is_dir() and not root.is_symlink()
    result={}
    for path in sorted(root.rglob('*')):
        name=str(path.relative_to(root))
        if path.is_symlink():
            assert str(path) in links and os.readlink(path)==links[str(path)]
            result[name]={'type':'symlink','target':os.readlink(path)}
        elif path.is_file():
            result[name]={'type':'file','size_bytes':path.stat().st_size,'sha256':sha(path)}
        else:
            assert path.is_dir()
            result[name]={'type':'directory'}
    return result


def verify(binding):
    assert binding['status']=='FROZEN_ROOT_APPROVED' and os.uname().nodename.split('.')[0]=='cvml04'
    assert sys.flags.isolated and sys.dont_write_bytecode and not os.environ.get('PYTHONPATH','')
    assert Path(sys.executable).resolve()==Path(binding['python']).resolve()
    assert binding['files'] and binding['trees']
    for path,digest in binding['files'].items():
        assert not Path(path).is_symlink() and Path(path).is_file() and sha(path)==digest,path
    for root,expected in binding['trees'].items():
        assert tree(root,binding.get('runtime_links',{}))==expected,root
    assert str(Path(binding['python']).resolve()) in binding['files']
    for path in (binding['environment_identity_receipt'],binding['runner'],binding['adapter'],
                 str(Path(binding['product_source'])/'policy_doctor/training_guard.py')):
        assert path in binding['files']
    for root in (binding['native_source'],binding['dataset'],binding['processor_assets'],*binding['checkpoints'].values()):
        assert root in binding['trees']


def audit(event,arguments):
    if event in ('socket.connect','socket.getaddrinfo','socket.gethostbyname','socket.sendto'):
        raise RuntimeError('Pilot uses local pinned inputs; Python network denied')


def summary(value):
    import torch

    if isinstance(value,torch.Tensor):
        actual=value.detach().cpu().contiguous()
        raw=actual.reshape(-1).view(torch.uint8).numpy().tobytes()
        return {'kind':'tensor','shape':list(actual.shape),'dtype':str(actual.dtype),
                'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    if isinstance(value,Mapping):
        return {str(key):summary(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):
        return [summary(item) for item in value]
    assert value is None or isinstance(value,(str,bool,int,float)),type(value)
    return value


def array_metadata(value):
    import numpy as np

    if isinstance(value,np.ndarray):
        return {'shape':list(value.shape),'dtype':str(value.dtype),
                'sha256':hashlib.sha256(value.tobytes()).hexdigest()}
    if isinstance(value,dict):
        return {str(key):array_metadata(item) for key,item in value.items()}
    return value


def seed():
    import numpy as np
    import torch

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


def owned_checkpoints(b,out):
    root=out/'checkpoints'
    newdir(root)
    original=load(Path(b['checkpoints']['baseline'])/'processor_config.json')
    false=load(Path(b['checkpoints']['relative_false'])/'processor_config.json')
    comparable=copy.deepcopy(false)
    comparable['processor_kwargs']['use_relative_action']=True
    assert comparable==original
    results={}
    for name in ('baseline','relative_false','inactive_q01'):
        target=root/name
        newdir(target)
        config=copy.deepcopy(original)
        kwargs=config['processor_kwargs']
        kwargs.update(model_name=b['processor_assets'],image_target_size=[256,256],image_crop_size=[230,230])
        kwargs['use_relative_action']=name!='relative_false'
        write(target/'processor_config.json',config)
        source=Path(b['checkpoints']['inactive_q01' if name=='inactive_q01' else 'baseline'])
        for filename in ('statistics.json','embodiment_id.json'):
            with (target/filename).open('xb') as stream:
                stream.write((source/filename).read_bytes())
        results[name]=target
    assert sha(results['baseline']/'statistics.json')==sha(results['relative_false']/'statistics.json')
    assert sha(results['baseline']/'statistics.json')!=sha(results['inactive_q01']/'statistics.json')
    # The correction is a new owned config with exactly False->True, never an
    # edited criterion or an overwritten fault/checkpoint artifact.
    corrected=root/'corrected'
    newdir(corrected)
    corrected_config=load(results['relative_false']/'processor_config.json')
    assert corrected_config['processor_kwargs']['use_relative_action'] is False
    corrected_config['processor_kwargs']['use_relative_action']=True
    assert corrected_config==load(results['baseline']/'processor_config.json')
    write(corrected/'processor_config.json',corrected_config)
    for filename in ('statistics.json','embodiment_id.json'):
        with (corrected/filename).open('xb') as stream:
            stream.write((results['relative_false']/filename).read_bytes())
    results['corrected']=corrected
    write(out/'owned-checkpoint-profile.json',{'model_name':b['processor_assets'],
        'public_processor':'Qwen/Qwen3-VL-2B-Instruct@89644892e4d85e24eaac8bacfd4f463576704203',
        'image_sizes_source':'pinned gr00t/configs/model/gr00t_n1d7.py54-55',
        'image_target_size':[256,256],'image_crop_size':[230,230],
        'correction_config_diff':{'pointer':'/processor_kwargs/use_relative_action','before':False,'after':True},
        'correction_stats_sha256':sha(corrected/'statistics.json'),
        'fault_stats_sha256':sha(results['relative_false']/'statistics.json'),
        'files':{str(path/filename):sha(path/filename) for path in results.values()
                 for filename in ('processor_config.json','statistics.json','embodiment_id.json')}})
    return results


def create_session(b,checkpoint,out,label,contract=None,*,expected_normalization=(False,False),max_samples=2):
    import numpy as np
    import torch
    from adapter import NativeSession, digest
    from gr00t.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset
    from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor

    declared=[json.loads(line) for line in (Path(b['dataset'])/'meta/episodes.jsonl').read_text().splitlines()]
    eligible=[int(item['length'])-49 for item in declared if int(item['length'])>=50]
    assert eligible and max(eligible)<=256,'Metadata sampling bound failed; preserve and revise protocol explicitly'
    seed()
    processor=Gr00tN1d7Processor.from_pretrained(checkpoint,
        transformers_loading_kwargs={'local_files_only':True,'trust_remote_code':False},local_files_only=True)
    assert type(processor).__name__=='Gr00tN1d7Processor'
    assert type(processor.collator).__name__=='Gr00tN1d7DataCollator'
    assert type(processor.processor).__name__==type(processor.collator.processor).__name__=='Qwen3VLProcessor'
    assert processor.training and processor.state_action_processor.training
    assert processor.use_relative_action==processor.state_action_processor.use_relative_action
    assert torch.get_default_dtype()==torch.float32 and torch.version.cuda is None
    assert processor.max_action_dim==processor.max_state_dim==64 and processor.max_action_horizon==50
    assert (processor.use_percentiles,processor.use_mean_std)==expected_normalization
    assert not any((processor.clip_outliers,processor.apply_sincos_state_encoding,
                    processor.exclude_state,processor.state_dropout_prob))
    tag=EmbodimentTag.UNITREE_G1
    profile=processor.modality_configs[tag.value]
    assert list(profile['action'].delta_indices)==list(range(50))
    assert list(profile['state'].delta_indices)==[0]
    seed()
    dataset=ShardedSingleStepDataset(b['dataset'],tag,profile,shard_size=1,
        episode_sampling_rate=1/256,seed=42,allow_padding=False)
    dataset.episode_loader.decoder_kwargs={'device':'cpu','num_ffmpeg_threads':1}
    lengths=[int(item['length']) for item in dataset.episode_loader.episodes_metadata]
    valid=[length-49 for length in lengths if length>=50]
    assert valid and max(valid)<=256 and all(int(n)==1 for n in dataset.shard_lengths)
    mixture=ShardedMixtureDataset([dataset],[1.0],processor,seed=42,training=True,
        num_shards_per_epoch=3,override_pretraining_statistics=False)
    assert processor.state_action_processor.statistics==load(checkpoint/'statistics.json')
    active=array_metadata(processor.state_action_processor.norm_params)
    config=load(checkpoint/'processor_config.json')
    processor_info={'checkpoint_config_sha256':sha(checkpoint/'processor_config.json'),
        'checkpoint_statistics_sha256':sha(checkpoint/'statistics.json'),
        'loaded_SAP_statistics_sha256':digest(processor.state_action_processor.statistics),
        'use_relative_action':processor.use_relative_action,'use_percentiles':processor.use_percentiles,
        'use_mean_std':processor.use_mean_std,
        'SAP_use_relative_action':processor.state_action_processor.use_relative_action,
        'active_normalization_parameters':active,'numeric_dtype':'torch.float32',
        'profile':config['processor_kwargs']['modality_configs'][tag.value],
        'native_processor_class':str(type(processor)),'native_collator_class':str(type(processor.collator))}
    dataset_files={name.removeprefix(b['dataset']+'/'):value for name,value in b['files'].items()
                   if name.startswith(b['dataset']+'/')}
    context={'files':b['files'],'dataset_content_sha256':digest(dataset_files),
        'processor':processor_info,'native_source':{'revision':b['native_revision'],
            'files':{name:value for name,value in b['files'].items() if name.startswith(b['native_source']+'/')}}}
    planned=[]
    for dataset_id,shard_id in mixture.shard_sampling_schedule:
        assert dataset_id==0
        planned.append({'shard':int(shard_id),'positions_steps':[
            [int(position),[int(step) for step in steps]] for position,steps in dataset.sharded_episodes[shard_id]]})
    write(out/(label+'-native-context.json'),{'processor':processor_info,'metadata_max_valid_steps':max(valid),
        'sampling_rate':1/256,'seed':42,'first3_native_schedule':planned,
        'classes':{name:{'file':inspect.getsourcefile(item),'sha256':sha(inspect.getsourcefile(item))}
                   for name,item in [('processor',type(processor)),('collator',type(processor.collator)),
                                     ('dataset',type(dataset)),('mixture',type(mixture))]},
        'versions':{name:importlib.metadata.version(name) for name in
            ('torch','torchvision','torchcodec','numpy','transformers')}})
    assert np.__version__=='1.26.4' and torch.__version__=='2.9.0+cpu'
    return NativeSession(dataset,mixture,processor.collator,context,contract,max_samples=max_samples)


def localize(report,observed):
    difference=report.get('first_difference')
    if not difference or difference.get('field')!='inputs.action' or 'coordinate' not in difference:
        return None
    _,time_index,column=difference['coordinate']
    reference=difference['provenance']
    fields=reference['source']['columns']
    matches=[field for field in fields if field['role']=='action'
             and field['packed_slice'][0]<=column<field['packed_slice'][1]]
    assert len(matches)==1
    field=matches[0]
    return {'native_tensor_coordinate':difference['coordinate'],'action_group':field['key'],
        'native_packed_slice':field['packed_slice'],'source_column':field['column'],
        'source_column_coordinate':field['source_slice'][0]+column-field['packed_slice'][0],
        'physical_action_row':reference['source']['rows']['action'][time_index],
        'stored_episode_index':reference['source']['stored_episode_index'],
        'declared_representation':field['declared_representation'],
        'declared_use_relative_action':reference['processor']['use_relative_action'],
        'observed_use_relative_action':observed['processor']['use_relative_action'],
        'observed_SAP_use_relative_action':observed['processor']['SAP_use_relative_action'],
        'claim':'Observed explicit loaded transform mismatch and final target difference; no mechanical/physical cause inferred'}


def consume(b,checkpoint,out,label,contract=None,count=2):
    from policy_doctor.training_guard import TargetContractViolation

    session=create_session(b,checkpoint,out,label,contract)
    references=[]
    reports=[]
    delivered=0
    try:
        for _index in range(count):
            try:
                batch=session.next()
            except TargetContractViolation as error:
                reports.append({'delivered':False,'guard':error.report,'batch':summary(session.tap.last_batch),
                    'loaded_transform_diagnostic':localize(error.report,session.tap.last_records[0].provenance),
                    'observed_provenance':[record.provenance for record in session.tap.last_records]})
                assert error.report['criterion_sha256']==contract.criterion_sha256
                break
            delivered+=1
            assert batch['inputs']['action'].shape==(1,50,64)
            assert batch['inputs']['action'].dtype.__str__()=='torch.float32'
            records=session.tap.last_records
            references.append((batch,[record.source_key for record in records],
                               [record.provenance for record in records]))
            reports.append({'delivered':True,'guard':session.guard.last_report if session.guard else None,
                'batch':summary(batch),'source_keys':[record.source_key for record in records]})
    finally:
        try:
            cleanup=session.close()
        finally:
            write(out/(label+'-cleanup.json'),session.cleanup or {'status':'FAILED_CLEANUP'})
    write(out/(label+'-reports.json'),{'delivered_batches':delivered,'requested_batches':count,
        'reports':reports,'cleanup':cleanup})
    return references,reports,delivered


def intent(relative):
    return {'criterion':'preserve these declared native reference action targets and masks exactly',
        'use_relative_action':relative,'use_percentiles':False,'clip_outliers':False,
        'dtype':'torch.float32','seed':42,'episode_sampling_rate':1/256,'reference_batches':2,
        'scope':'Covered declared source samples only; no historical/physical/model claim'}


def run(b,out,phase,contract_path,pin):
    from policy_doctor.training_guard import TargetContract

    checkpoints=owned_checkpoints(b,out)
    if phase=='readiness':
        refs,_,delivered=consume(b,checkpoints['baseline'],out,'readiness',count=1)
        assert delivered==1
        return {'status':'PASS_REAL_NATIVE_BATCH_READINESS','batch':summary(refs[0][0]),
                'source_keys':refs[0][1],'provenance':refs[0][2],'native_training_model_forward':False}
    if phase=='replay':
        assert contract_path and pin,'Fresh process requires retained pre-fault pin via explicit CLI argument'
        contract=TargetContract.load(contract_path,expected_sha256=pin)
        _,reports,delivered=consume(b,checkpoints['corrected'],out,'fresh-process-replay',contract)
        assert delivered==2 and all(item['guard']['status']=='pass' for item in reports)
        return {'status':'PASS_FRESH_PROCESS_SAME_CRITERION','criterion_sha256':pin,'delivered':delivered}
    refs,_,delivered=consume(b,checkpoints['baseline'],out,'reference')
    assert delivered==2 and len({key for _,keys,_ in refs for key in keys})==2
    contract=TargetContract.from_references(refs,intent=intent(True))
    path=out/'reference-contract.json'
    contract.save(path)
    pin=contract.criterion_sha256
    write(out/'pre-fault-reference-freeze.json',{'criterion_sha256':pin,'contract':str(path),
        'source_keys':[keys for _,keys,_ in refs],'core_module_sha256':sha(inspect.getsourcefile(TargetContract)),
        'claim':'Declared True recipe preservation; no independent historical or physical authentication'})
    contract=TargetContract.load(path,expected_sha256=pin)
    for batch,keys,_ in refs:
        assert contract.check(batch,keys)['status']=='pass'
    _,fault,blocked_deliveries=consume(b,checkpoints['relative_false'],out,'fault',contract)
    contradiction=[]
    for item in fault:
        report=item['guard']
        action_findings=[finding for finding in report['findings']
            if finding['code']=='target_values_differ' and finding['field']=='inputs.action'
            and finding['differing_values']>0]
        mask_findings=[finding for finding in report['findings']
            if finding['code']=='target_schema_mismatch' and finding['field']=='inputs.action_mask'
            and finding['expected_shape']==finding['observed_shape']==[50,64]
            and finding['expected_dtype']=='float64' and finding['observed_dtype']=='float32']
        diagnostic=item['loaded_transform_diagnostic']
        if (blocked_deliveries==0 and not item['delivered'] and report['status']=='error'
                and report['differing_target_values']>0
                and report['target_values_checked']==3200 and report['mask_values_checked']==0
                and report['finding_count']==2 and not report['findings_truncated']
                and len(action_findings)==len(mask_findings)==1
                and diagnostic['declared_use_relative_action'] is True
                and diagnostic['observed_use_relative_action'] is False
                and diagnostic['observed_SAP_use_relative_action'] is False
                and item['batch']['inputs']['action']['dtype']=='torch.float32'):
            contradiction.append(item)
    _,corrected,delivered=consume(b,checkpoints['corrected'],out,'corrected',TargetContract.load(path,expected_sha256=pin))
    assert delivered==2 and all(item['guard']['status']=='pass' for item in corrected)
    _,inactive,delivered=consume(b,checkpoints['inactive_q01'],out,'inactive-q01',TargetContract.load(path,expected_sha256=pin))
    assert delivered==2 and all(item['guard']['status']=='pass' for item in inactive)
    false_refs,_,delivered=consume(b,checkpoints['relative_false'],out,'intentional-false-reference')
    assert delivered==2
    lawful=TargetContract.from_references(false_refs,intent=intent(False))
    false_path=out/'intentional-false-contract.json'
    lawful.save(false_path)
    assert lawful.criterion_sha256!=pin
    _,lawful_reports,delivered=consume(b,checkpoints['relative_false'],out,'intentional-false',
                                     TargetContract.load(false_path,expected_sha256=lawful.criterion_sha256))
    assert delivered==2 and all(item['guard']['status']=='pass' for item in lawful_reports)
    assert sha(path)==pin
    return {'status':'PASS_REAL_NATIVE_GUARD_LOOP' if contradiction else 'UNPROVED_DECLARED_COMBINED_NATIVE_CONSEQUENCE',
        'criterion_sha256':pin,'contract':str(path),'fault_delivered_before_block':blocked_deliveries,
        'fault_first_difference':contradiction[0]['guard']['first_difference'] if contradiction else None,
        'fault_findings':contradiction[0]['guard']['findings'] if contradiction else None,
        'fault_mask_values_compared':False,
        'fault_mask_scope':'Independent native mask dtype mismatch; mask values not compared across dtype',
        'loaded_transform_diagnostic':contradiction[0]['loaded_transform_diagnostic'] if contradiction else None,
        'corrected_batches':2,'later_second_reference_passed':True,'inactive_identity_control_passed':True,
        'intentional_false_criterion':lawful.criterion_sha256,'intentional_false_control_passed':True,
        'fresh_process_replay_pending':True,'scope':'Declared native target/mask preservation; no model/training/physical outcome'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--bindings',required=True)
    parser.add_argument('--bindings-sha256',required=True)
    parser.add_argument('--phase',choices=('readiness','loop','replay'),required=True)
    parser.add_argument('--contract')
    parser.add_argument('--criterion-sha256')
    args=parser.parse_args()
    assert sha(args.bindings)==args.bindings_sha256
    b=load(args.bindings)
    verify(b)
    assert sys.pycache_prefix==b['pycache_prefixes'][args.phase] and not os.path.lexists(sys.pycache_prefix)
    sys.addaudithook(audit)
    # All three source roots are explicit trusted code, bound before import.
    sys.path[:0]=[b['product_source'],b['native_source'],str(Path(__file__).parent)]
    import torch

    torch.set_num_threads(1)
    base=Path(b['output_root'])
    if not base.exists():
        newdir(base)
    out=base/args.phase
    newdir(out)
    result=None
    try:
        result=run(b,out,args.phase,args.contract,args.criterion_sha256)
    except BaseException as error:
        result={'status':'FAILED_PRESERVED','type':type(error).__name__,'message':str(error),
                'traceback':traceback.format_exc()}
    finally:
        verify(b)
        if result is not None:
            result['binding_sha256']=args.bindings_sha256
            write(out/'result.json',result)
    print(json.dumps(result,sort_keys=True))
    return 0 if result and result['status'].startswith('PASS_') else 2


if __name__=='__main__':
    raise SystemExit(main())
