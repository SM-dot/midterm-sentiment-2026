"""Small HTTP helper with exponential back-off; all collectors go through it."""
import time, random, requests

UA = {"User-Agent": "Mozilla/5.0 (academic research; midterm-sentiment-2026)"}
_session = requests.Session()
_session.headers.update(UA)


def get(url, params=None, tries=6, timeout=30, base_sleep=2.0, ok=(200,)):
    last = None
    for k in range(tries):
        try:
            r = _session.get(url, params=params, timeout=timeout)
            if r.status_code in ok:
                return r
            last = f"HTTP {r.status_code}: {r.text[:120]}"
            if r.status_code in (400, 401, 403, 404):
                return r  # not retryable
        except requests.RequestException as e:
            last = repr(e)
        time.sleep(base_sleep * (2 ** k) + random.random())
    raise RuntimeError(f"GET failed after {tries} tries: {url} {params} :: {last}")
