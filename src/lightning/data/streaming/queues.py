# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing
import os
import pickle
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Callable, Dict, Optional
from urllib.parse import urljoin

import requests
from pydantic import BaseModel
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class BaseQueue(ABC):
    """Base Queue class that has a similar API to the Queue class in python."""

    @abstractmethod
    def __init__(self, name: str, default_timeout: float):
        self.name = name
        self.default_timeout = default_timeout

    @abstractmethod
    def put(self, item: Any) -> None:
        pass

    @abstractmethod
    def get(self, timeout: Optional[float] = None) -> Any:
        """Returns the left most element of the queue.

        Parameters
        ----------
        timeout:
            Read timeout in seconds, in case of input timeout is 0, the `self.default_timeout` is used.
            A timeout of None can be used to block indefinitely.

        """
        pass

    @property
    def is_running(self) -> bool:
        """Returns True if the queue is running, False otherwise.

        Child classes should override this property and implement custom logic as required

        """
        return True


class MultiProcessQueue(BaseQueue):
    def __init__(self, name: str, default_timeout: float) -> None:
        self.name = name
        self.default_timeout = default_timeout
        context = multiprocessing.get_context("spawn")
        self.queue = context.Queue()

    def put(self, item: Any) -> None:
        self.queue.put(item)

    def get(self, timeout: Optional[float] = None) -> Any:
        if timeout == 0:
            timeout = self.default_timeout
        return self.queue.get(timeout=timeout, block=(timeout is None))


_CONNECTION_RETRY_TOTAL = 2880
_CONNECTION_RETRY_BACKOFF_FACTOR = 0.5
_DEFAULT_REQUEST_TIMEOUT = 30  # seconds


class CustomRetryAdapter(HTTPAdapter):
    def __init__(self, *args: Any, **kwargs: Any):
        self.timeout = kwargs.pop("timeout", _DEFAULT_REQUEST_TIMEOUT)
        super().__init__(*args, **kwargs)

    def send(self, request, **kwargs: Any):
        kwargs["timeout"] = kwargs.get("timeout", self.timeout)
        return super().send(request, **kwargs)


def _http_method_logger_wrapper(func: Callable) -> Callable:
    """Returns the function decorated by a wrapper that logs the message using the `log_function` hook."""

    @wraps(func)
    def wrapped(self: "HTTPClient", *args: Any, **kwargs: Any) -> Any:
        message = f"HTTPClient: Method: {func.__name__.upper()}, Path: {args[0]}\n"
        message += f"      Base URL: {self.base_url}\n"
        params = kwargs.get("query_params", {})
        if params:
            message += f"      Params: {params}\n"
        resp: requests.Response = func(self, *args, **kwargs)
        message += f"      Response: {resp.status_code} {resp.reason}"
        self.log_function(message)
        return resp

    return wrapped


def _response(r, *args: Any, **kwargs: Any):
    return r.raise_for_status()


class HTTPClient:
    """A wrapper class around the requests library which handles chores like logging, retries, and timeouts
    automatically."""

    def __init__(
        self, base_url: str, auth_token: Optional[str] = None, log_callback: Optional[Callable] = None
    ) -> None:
        self.base_url = base_url
        retry_strategy = Retry(
            # wait time between retries increases exponentially according to: backoff_factor * (2 ** (retry - 1))
            # but the the maximum wait time is 120 secs. By setting a large value (2880), we'll make sure clients
            # are going to be alive for a very long time (~ 4 days) but retries every 120 seconds
            total=_CONNECTION_RETRY_TOTAL,
            backoff_factor=_CONNECTION_RETRY_BACKOFF_FACTOR,
            status_forcelist=[
                408,  # Request Timeout
                429,  # Too Many Requests
                500,  # Internal Server Error
                502,  # Bad Gateway
                503,  # Service Unavailable
                504,  # Gateway Timeout
            ],
        )
        adapter = CustomRetryAdapter(max_retries=retry_strategy, timeout=_DEFAULT_REQUEST_TIMEOUT)
        self.session = requests.Session()

        self.session.hooks = {"response": _response}

        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        if auth_token:
            self.session.headers.update({"Authorization": f"Bearer {auth_token}"})

        self.log_function = log_callback or self.log_function

    @_http_method_logger_wrapper
    def get(self, path: str):
        url = urljoin(self.base_url, path)
        return self.session.get(url)

    @_http_method_logger_wrapper
    def post(self, path: str, *, query_params: Optional[Dict] = None, data: Optional[bytes] = None, json: Any = None):
        url = urljoin(self.base_url, path)
        return self.session.post(url, data=data, params=query_params, json=json)

    @_http_method_logger_wrapper
    def delete(self, path: str):
        url = urljoin(self.base_url, path)
        return self.session.delete(url)

    def log_function(self, message: str, *args, **kwargs: Any):
        """This function is used to log the messages in the client, it can be overridden by caller to customise the
        logging logic.

        We enabled customisation here instead of just using `logger.debug` because HTTP logging can be very noisy, but
        it is crucial for finding bugs when we have them

        """
        pass


def _get_node_rank() -> int:
    """Returns the current node rank of the instance."""
    return int(os.getenv("DATA_OPTIMIZER_NODE_RANK", 0))


class HTTPQueue(BaseQueue):
    def __init__(self):
        self.client: Optional[HTTPClient] = None

    def get(self, timeout: Optional[float] = None) -> Any:
        raise NotImplementedError

    def put(self, items: Dict[int, Any]) -> Any:
        lightning_app_state_url = os.getenv("LIGHTNING_APP_STATE_URL")
        if lightning_app_state_url is None:
            raise RuntimeError("This shouldn't have happened.")
        if self.client is None:
            self.client = HTTPClient(lightning_app_state_url)
        return self._put(items)

    def _put(self, items: Dict[int, Any]) -> Any:
        json = {
            "node_rank": _get_node_rank(),
            "data": {k: pickle.dumps(v, 0).decode() for k, v in items.items()},
        }
        return self.client.post("/put", json=json)


class BroadcastInput(BaseModel):
    key: str
    value: str


class Broadcaster:
    def __init__(self):
        self.client: Optional[HTTPClient] = None

    def broadcast(self, key: str, obj: Any) -> Any:
        lightning_app_state_url = os.getenv("LIGHTNING_APP_STATE_URL")
        if lightning_app_state_url is None:
            return obj
        if self.client is None:
            self.client = HTTPClient(lightning_app_state_url)
        return self._broadcast(key, obj)

    def _broadcast(self, key: str, obj: Any) -> Any:
        json = BroadcastInput(key=key, value=pickle.dumps(obj, 0).decode()).model_dump()
        resp = self.client.post("/broadcast", json=json)
        if resp.status_code != 200:
            raise RuntimeError("Failed to broadcast value.")
        return pickle.loads(bytes(resp.json()["value"], "utf-8"))