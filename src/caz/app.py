import asyncio
import logging
from datetime import datetime

from caz.views.chess_main import Application


def launch() -> None:
    # Generate a timestamp string
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.basicConfig(
        filename=f"app_{timestamp}.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    try:
        loop = asyncio.get_event_loop()
        app = Application(loop)
        asyncio.run(app.show_async())
    except:
        logging.exception("")


if __name__ == "__main__":
    launch()
