
class UserError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message
        self.kvs = {}

    def set_http_status(self, http_code):
        self.http_code = http_code
        return self

    def set_kvs(self, kvs: dict):
        self.kvs.update(kvs)
        return self

    def to_dict(self):
        error_dict = {
            "error": self.message
        }
        if hasattr(self, "kvs"):
            error_dict.update(self.kvs)
        return error_dict