"""Bounded static selection of one simple launch command; never a shell interpreter."""
from __future__ import annotations

import re
import shlex
from pathlib import PurePosixPath

MAX_COMMANDS = 16
MAX_TOKENS = 256


def _segments(command):
    if type(command) is not str or not command or len(command) > 8192:
        return None
    parts, start, quote, escaped, index = [], 0, None, False, 0
    while index < len(command):
        char = command[index]
        if escaped:
            escaped = False
        elif char == '\\' and quote != "'":
            escaped = True
        elif quote:
            if char == quote:
                quote = None
            elif quote == '"' and char in '$`':
                return None
        elif char in "'\"":
            quote = char
        elif char in '$`(){}<>#~*?[]':
            return None
        elif char in ';&|':
            operator = char
            if index + 1 < len(command) and command[index + 1] == char:
                operator += char
            if operator not in (';', '&&', '||'):
                return None
            segment = command[start:index].strip()
            if not segment or len(parts) >= MAX_COMMANDS:
                return None
            parts.append(segment)
            index += len(operator) - 1
            start = index + 1
        index += 1
    if quote or escaped:
        return None
    tail = command[start:].strip()
    if tail:
        parts.append(tail)
    elif not command.rstrip().endswith(';'):
        return None
    if not parts or len(parts) > MAX_COMMANDS:
        return None
    return parts


def _script(value):
    return 'script:' + str(PurePosixPath(value))


def _launch(tokens, depth=0):
    if not tokens or depth > 3:
        return None
    while tokens and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*=.*', tokens[0]):
        tokens = tokens[1:]
    if not tokens:
        return None
    executable = PurePosixPath(tokens[0]).name
    tail = tokens[1:]
    if executable == 'env':
        return _launch(tail, depth + 1) if tail and not tail[0].startswith('-') else None
    if executable == 'uv':
        if not tail or tail[0] != 'run':
            return None
        tail = tail[1:]
        while tail and tail[0].startswith('-'):
            key, equals, _ = tail[0].partition('=')
            if key == '--':
                tail = tail[1:]
                break
            if key in ('--no-sync', '--offline', '--frozen', '--locked', '--no-project') and not equals:
                tail = tail[1:]
            elif key in ('--project', '--directory', '--python'):
                if not equals and (len(tail) < 2 or tail[1].startswith('-')):
                    return None
                tail = tail[1 if equals else 2:]
            else:
                return None
        return _launch(tail, depth + 1)
    if re.fullmatch(r'python(?:[0-9]+(?:\.[0-9]+)*)?', executable):
        while tail and tail[0].startswith('-'):
            option = tail[0]
            if option == '-m':
                if len(tail) < 2:
                    return None
                if tail[1] == 'torch.distributed.run':
                    return _launch(['torchrun', *tail[2:]], depth + 1)
                return ('module:' + tail[1], tail[2:])
            if option in ('-u', '-B', '-O', '-OO', '-s', '-E', '-I', '-q'):
                tail = tail[1:]
            elif option in ('-W', '-X') and len(tail) >= 2:
                tail = tail[2:]
            else:
                return None
        return (_script(tail[0]), tail[1:]) if tail else None
    if executable == 'torchrun':
        module = False
        values = {'--nproc-per-node', '--nnodes', '--node-rank', '--master-addr', '--master-port',
                  '--rdzv-backend', '--rdzv-endpoint', '--rdzv-id', '--max-restarts', '--monitor-interval'}
        while tail and tail[0].startswith('-'):
            key, equals, _ = tail[0].partition('=')
            key = key.replace('_', '-')
            if key == '--':
                tail = tail[1:]
                break
            if key in ('-m', '--module') and not equals:
                module = True
                tail = tail[1:]
            elif key in ('--standalone', '--no-python', '--run-path') and not equals:
                tail = tail[1:]
            elif key in values:
                if not equals and (len(tail) < 2 or tail[1].startswith('-')):
                    return None
                tail = tail[1 if equals else 2:]
            else:
                return None
        if not tail:
            return None
        return (('module:' + tail[0]) if module else _script(tail[0]), tail[1:])
    if executable in ('if', 'then', 'else', 'fi', 'for', 'while', 'until', 'case', 'do', 'done', 'function', 'eval', 'sh', 'bash', 'zsh',
                      '!', 'time', 'command', 'exec', 'nohup', 'nice', 'sudo', 'conda', 'poetry', 'hatch', 'pdm', 'pipenv',
                      'xargs', 'parallel', 'sbatch', 'srun', 'mpirun', 'mpiexec'):
        return None
    if any(token.endswith('.py') for token in tail):
        return None  # A possible unmodeled wrapper, never the first arbitrary .py argument.
    return ((_script(tokens[0]) if tokens[0].endswith('.py') else 'entrypoint:' + executable), tail)


def _flags(arguments):
    flags = {}
    for index, token in enumerate(arguments):
        if token == '--':
            break
        if not token.startswith('--'):
            continue
        key, equals, value = token.partition('=')
        if not equals:
            value = arguments[index + 1] if index + 1 < len(arguments) and not arguments[index + 1].startswith('--') else True
        flags.setdefault(key, []).append(value)
    return flags


def select_command(command, targets=None):
    segments = _segments(command)
    result = {'state': 'unknown', 'flags': None, 'reason': 'No single supported applicable command was selected.'}
    if segments is None:
        return result
    commands, count = [], 0
    for segment in segments:
        try:
            tokens = shlex.split(segment, posix=True)
        except ValueError:
            return result
        count += len(tokens)
        if count > MAX_TOKENS:
            return result
        selected = _launch(tokens)
        if selected is None:
            return result
        # Explicit flags survive supported wrappers, but an absence default
        # needs a single direct invocation whose cwd/environment was declared.
        # Do not infer those effects from preparation commands or uv/env.
        direct = (len(segments) == 1 and tokens
                  and PurePosixPath(tokens[0]).name not in ('uv', 'env')
                  and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*=.*', tokens[0]) is None)
        commands.append((*selected, bool(direct)))
    selected = [item for item in commands if targets is None or item[0] in targets]
    if not selected and targets is not None:
        if len(commands) != 1:
            return result
        return {'state': 'absent', 'flags': None, 'reason': 'No parsed command selects the entry launch_target.'}
    if len(selected) != 1:
        return result
    selector, arguments, source_binding_safe = selected[0]
    return {'state': 'selected', 'selector': selector, 'flags': _flags(arguments),
            'source_binding_safe': source_binding_safe,
            'reason': 'One statically declared simple command selected; execution is not attested.'}
