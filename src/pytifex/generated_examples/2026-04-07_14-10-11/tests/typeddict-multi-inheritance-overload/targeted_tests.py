"""
Targeted Test Suite — Pattern-Based Type Error Detection

Source: typeddict-multi-inheritance-overload.py
Patterns detected: 1
    - decorator_signature (2 tests)
Test cases generated: 2
"""

# --- Original source ---

from typing import TypedDict, NotRequired, Required, overload

class BaseOptions(TypedDict, total=False):
    debug: bool
    verbose: NotRequired[bool]

class CommonSettings(TypedDict, total=True):
    timeout: int

class RequestOptions(BaseOptions, CommonSettings):
    url: Required[str]
    method: NotRequired[str]

class WorkerOptions(BaseOptions, total=False):
    concurrency: int
    queue_name: Required[str]

@overload
def process_task(options: RequestOptions) -> str: ...
@overload
def process_task(options: WorkerOptions) -> str: ...
def process_task(options: RequestOptions | WorkerOptions) -> str:
    if "url" in options and "method" in options:
        # RequestOptions path
        return f"Processing request to {options['url']} with method {options.get('method', 'GET')}, debug={options.get('debug', False)}"
    elif "concurrency" in options and "queue_name" in options:
        # WorkerOptions path
        return f"Processing worker task on queue {options['queue_name']} with concurrency {options['concurrency']}, verbose={options.get('verbose', False)}"
    else:
        raise ValueError("Unknown options type")

if __name__ == "__main__":
    # mypy's error reporting for overloads combined with complex TypedDict inheritance
    # (especially mixing total=True/False and Required/NotRequired) can be tricky.
    # The original issue was about incorrect method names in overload notes.
    # This checks if the checker correctly identifies which overload applies and if it
    # correctly validates the required/notrequired fields based on inheritance.
    req_opts: RequestOptions = {"url": "http://example.com", "timeout": 30, "debug": True}
    print(process_task(req_opts))

    worker_opts: WorkerOptions = {"queue_name": "high_prio", "concurrency": 5, "verbose": True}
    print(process_task(worker_opts))

    # This should be valid - `debug` is inherited from total=False BaseOptions.
    req_opts_minimal: RequestOptions = {"url": "http://api.com", "timeout": 10}
    print(process_task(req_opts_minimal))

    # This should error for `RequestOptions` (missing `timeout`).
    # invalid_req: RequestOptions = {"url": "http://bad.com"}
    # print(process_task(invalid_req))

    # This should error for `WorkerOptions` (missing `queue_name`).
    # invalid_worker: WorkerOptions = {"concurrency": 2}
    # print(process_task(invalid_worker))

# --- Test infrastructure ---
BUGS = []
_SOURCE_LINE_OFFSET = 11

# --- Test cases ---

def test_process_task_decorated_callable():
    """Verify decorated function process_task is callable."""
    if not callable(process_task):
        BUGS.append({"line": 19, "type": "TypeError", "error": "process_task is not callable after decoration", "test": "decorated_callable"})


def test_process_task_decorated_callable():
    """Verify decorated function process_task is callable."""
    if not callable(process_task):
        BUGS.append({"line": 21, "type": "TypeError", "error": "process_task is not callable after decoration", "test": "decorated_callable"})


# --- Runner ---
if __name__ == "__main__":
    import sys
    _test_fns = [(name, fn) for name, fn in list(globals().items()) if name.startswith("test_") and callable(fn)]
    print(f"Running {len(_test_fns)} targeted tests...")
    _passed = 0
    _failed = 0
    for _name, _fn in _test_fns:
        try:
            _fn()
            _passed += 1
        except Exception as _e:
            _failed += 1
    print(f"Passed: {_passed}, Failed: {_failed}, Bugs found: {len(BUGS)}")
    for _bug in BUGS:
        print(f"  BUG L{_bug['line']} [{_bug['type']}] {_bug['error']}")
