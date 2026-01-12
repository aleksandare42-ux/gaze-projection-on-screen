import asyncio
import threading

def run_async(self, coro):
    if not hasattr(self, "_loop"):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        threading.Thread(target=self._loop.run_forever, daemon=True).start()
    asyncio.run_coroutine_threadsafe(coro, self._loop)

