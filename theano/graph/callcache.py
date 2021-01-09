import logging
import pickle


_logger = logging.getLogger("theano.graph.callcache")


class CallCache:
    def __init__(self, filename=None):
        self.filename = filename
        try:
            if filename is None:
                raise OSError("bad filename")  # just goes to except
            with open(filename) as f:
                self.cache = pickle.load(f)
        except OSError:
            self.cache = {}

    def persist(self, filename=None):
        """
        Cache "filename" as a pickle file
        """
        if filename is None:
            filename = self.filename
        with open(filename, "w") as f:
            pickle.dump(self.cache, f)

    def call(self, fn, args=(), key=None):
        """
        Retrieve item from the cache(if available)
        based on a key

        Parameters:
        ----------
        key
            parameter to retrieve cache item
        fn,args
            key to retrieve if "key" is None
        """
        if key is None:
            key = (fn, tuple(args))
        if key not in self.cache:
            _logger.debug("cache miss %i", len(self.cache))
            self.cache[key] = fn(*args)
        else:
            _logger.debug("cache hit %i", len(self.cache))
        return self.cache[key]

    def __del__(self):
        try:
            if self.filename:
                self.persist()
        except Exception as e:
            _logger.error("persist failed %s %s", self.filename, e)
