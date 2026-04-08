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