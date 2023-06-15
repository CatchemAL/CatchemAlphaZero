import asyncio
import sys
import time
from asyncio import Event


async def main():
    ponder = True  # smelly global

    async def ainput() -> str:
        await asyncio.to_thread(sys.stdout.write, f"Please enter your move: ")
        response = await asyncio.to_thread(sys.stdin.readline)
        return response.rstrip("\n")

    async def ponder_old() -> int:
        i = 1
        print()
        while ponder and i <= 150:
            await asyncio.sleep(0.1)
            print(f"\n{i}", end=" ")
            i += 1

        return i

    async def ponder_async(event: Event) -> int:
        i = 1
        print()
        while i <= 150 and not event.is_set():
            await asyncio.sleep(0.1)
            print(f"\n{i}", end=" ")
            i += 1

        return i

    async def ponder_halfsync() -> int:
        i = 1
        print()
        while ponder and i <= 150:
            await asyncio.sleep(0)
            time.sleep(0.1)
            print(f"\n{i}", end=" ")
            i += 1

        return i

    async def ponder_async(event: Event) -> int:
        i = 1
        print()
        while i <= 150 and not event.is_set():
            await asyncio.sleep(0)

            # Simulate CPU bound work with blocking call
            time.sleep(0.1)
            print(f"\n{i}", end=" ")
            i += 1

        return i

    event = asyncio.Event()

    input_coroutine = ainput()
    print_coroutine = ponder_async(event)

    get_player_move_task = asyncio.create_task(input_coroutine)
    ponder_task = asyncio.create_task(print_coroutine)

    tasks = [get_player_move_task, ponder_task]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

    if get_player_move_task in done:
        ponder = False
        event.set()
        reached = await ponder_task
        user_input = await get_player_move_task
        print(f"\n Thanks - your move was: {user_input}")
        print(f"Whilst you were thinking, I also calculated {reached} steps ahead 😇")
    else:
        print("I've maxed out my calculation whilst you were thinking. Are you getting scared???")
        reached = await ponder_task
        print(f"Whilst you were thinking, I also calculated {reached} steps ahead 😇")
        print(f"It's still your turn btw...")
        user_input = await get_player_move_task
        print(f"\n Thanks - your move was: {user_input}")


asyncio.run(main())
