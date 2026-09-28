import json
from urllib.request import Request, urlopen

from embodied_data_lab.operator_web import OperatorWebServer


def test_operator_web_serves_page_state_and_commands():
    server = OperatorWebServer(port=0)
    server.start()
    base = f"http://127.0.0.1:{server.port}"
    try:
        page = urlopen(base + "/", timeout=2).read().decode("utf-8")
        assert "front presentation robot camera views" in page
        assert "saved as an MP4 during recording" in page
        assert "Complete 3 recordings in order" in page
        assert "Controls from behind the robot" in page
        assert "Restart current take" in page

        state = json.loads(urlopen(base + "/state", timeout=2).read())
        assert state["message"] == "Loading simulator..."

        request = Request(base + "/key", data=b"w_down", method="POST")
        assert urlopen(request, timeout=2).status == 204
        assert server.next_command() == "w_down"

        request = Request(base + "/key", data=b"w_up", method="POST")
        assert urlopen(request, timeout=2).status == 204
        assert server.next_command() == "w_up"
    finally:
        server.stop()
