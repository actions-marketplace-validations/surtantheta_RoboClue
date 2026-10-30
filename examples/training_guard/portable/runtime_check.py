"""Check the actual decoder before declaring the private preview installed."""
import ctypes
import importlib.util
import json
import sys
from pathlib import Path


def main():
    try:
        import torch
        import torchcodec
        from torchcodec.decoders import VideoDecoder

        if torch.__version__ != '2.9.0+cpu' or torchcodec.__version__ != '0.8.0':
            raise ValueError('This preview requires its pinned CPU Torch/TorchCodec versions')
        mappings = Path('/proc/self/maps').read_text()
        if 'libtorchcodec_core4.so' not in mappings or 'libavcodec.so.58' not in mappings:
            raise ValueError('This preview qualifies FFmpeg 4; the active decoder uses a different runtime')
        print(json.dumps({'status': 'decoder_ready', 'torch': torch.__version__,
                          'torchcodec': torchcodec.__version__, 'ffmpeg_major': 4,
                          'decoder_class': VideoDecoder.__name__}))
        return 0
    except Exception as error:
        diagnostics = []
        spec = importlib.util.find_spec('torchcodec')
        if spec is not None and spec.submodule_search_locations:
            root = Path(next(iter(spec.submodule_search_locations)))
            for name in ('libtorchcodec_core4.so', 'libtorchcodec_custom_ops4.so', 'libtorchcodec_pybind_ops4.so'):
                try:
                    ctypes.CDLL(str(root / name))
                except OSError as library_error:
                    diagnostics.append({'library': name, 'loader_error': str(library_error)})
                    break
        print(json.dumps({'status': 'decoder_not_ready', 'message': str(error),
                          'loader_diagnostics': diagnostics,
                          'next_step': 'Install the FFmpeg 4 shared runtime and make libpython3.12.so.1.0 discoverable. '
                                       'For a standalone Python distribution, include its lib directory in LD_LIBRARY_PATH. '
                                       'Installation has not qualified the startup workflow.'}), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
