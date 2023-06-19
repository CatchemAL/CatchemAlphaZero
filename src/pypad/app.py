import asyncio

from pypad.views.chess_main import Application


def launch() -> None:
    loop = asyncio.get_event_loop()
    app = Application(loop)
    asyncio.run(app.show_async())


if __name__ == "__main__":
    launch()
