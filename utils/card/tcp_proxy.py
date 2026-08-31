#!/usr/bin/env python3
"""Small fixed-target TCP relay for an isolated Docker service."""

import argparse
import asyncio


async def relay(reader, writer):
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    finally:
        try:
            writer.write_eof()
            await writer.drain()
        except (AttributeError, ConnectionError, OSError):
            writer.close()


async def handle(client_reader, client_writer, target_host, target_port):
    try:
        target_reader, target_writer = await asyncio.open_connection(target_host, target_port)
    except (ConnectionError, OSError):
        client_writer.close()
        await client_writer.wait_closed()
        return
    await asyncio.gather(
        relay(client_reader, target_writer),
        relay(target_reader, client_writer),
        return_exceptions=True,
    )
    target_writer.close()
    client_writer.close()
    await asyncio.gather(
        target_writer.wait_closed(), client_writer.wait_closed(), return_exceptions=True,
    )


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target_host")
    parser.add_argument("target_port", type=int)
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=8000)
    args = parser.parse_args()
    server = await asyncio.start_server(
        lambda reader, writer: handle(
            reader, writer, args.target_host, args.target_port,
        ),
        args.listen_host,
        args.listen_port,
    )
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
