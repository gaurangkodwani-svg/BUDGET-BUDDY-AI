import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from streamlit.web import cli

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "app.py", "--server.headless=true", "--server.enableCORS=false", "--server.enableXsrfProtection=false", "--server.fileWatcherType=none"]
    cli.main()
