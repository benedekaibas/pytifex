from typing import TypedDict, Unpack, NotRequired, Required, Any

class BaseConfig(TypedDict, total=False):
    debug: bool
    log_level: NotRequired[str]

class ServerConfig(BaseConfig, total=True, extra_items=Any):
    host: Required[str]
    port: int

class ClientConfig(BaseConfig, total=False):
    timeout: int
    user_agent: NotRequired[str]

def process_server_request(**kwargs: Unpack[ServerConfig]):
    print(f"Processing server request: {kwargs}")
    assert 'host' in kwargs # Should be required by ServerConfig
    assert 'port' in kwargs # Should be required by ServerConfig
    if 'debug' in kwargs:
        assert isinstance(kwargs['debug'], bool)
    if 'log_level' in kwargs:
        assert isinstance(kwargs['log_level'], str)

def process_client_request(**kwargs: Unpack[ClientConfig]):
    print(f"Processing client request: {kwargs}")
    if 'timeout' in kwargs:
        assert isinstance(kwargs['timeout'], int)
    if 'user_agent' in kwargs:
        assert isinstance(kwargs['user_agent'], str)
    # mypy might struggle with 'debug' not being explicitly NotRequired in ClientConfig
    # but inherited from total=False BaseConfig.
    if 'debug' in kwargs:
        assert isinstance(kwargs['debug'], bool)

if __name__ == "__main__":
    process_server_request(host="localhost", port=8080, debug=True)
    process_server_request(host="127.0.0.1", port=5000) # log_level and debug are optional

    # Should be valid according to ClientConfig (timeout required if present, debug from BaseConfig)
    process_client_request(timeout=10, debug=False)
    process_client_request(timeout=5, user_agent="test-client")
    process_client_request() # All fields optional for ClientConfig

    # This call should ideally error for ServerConfig due to missing 'host'
    # func_that_should_error(**{'port': 80})
    # mypy and pyright can diverge on error reporting for unpacked TypedDicts with inheritance.