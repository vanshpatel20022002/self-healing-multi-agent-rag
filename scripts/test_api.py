import asyncio
import sys
from pathlib import Path

from httpx import ASGITransport, AsyncClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.main import app  # noqa: E402


async def main() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        print(response.status_code, response.json())


if __name__ == "__main__":
    asyncio.run(main())
