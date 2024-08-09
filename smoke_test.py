import unittest
import requests
import time


class SmokeTest(unittest.TestCase):
    def _waitForStartup(self):
        url = "http://localhost:8002/.well-known/ready"

        for i in range(0, 100):
            try:
                res = requests.get(url)
                if res.status_code == 204:
                    return
                else:
                    raise Exception("status code is {}".format(res.status_code))
            except Exception as e:
                print("Attempt {}: {}".format(i, e))
                time.sleep(1)

        raise Exception("did not start up")

    def testVectorizing(self):
        self._waitForStartup()
        url = "http://localhost:8002/vectors/"
        req_body = {
            "id": "test",
            "image": "/9j/4AAQSkZJRgABAQEASABIAAD/4QpKRXhpZgAASUkqAAgAAAAGABoBBQABAAAAVgAAABsBBQABAAAAXgAAACgBAwABAAAAAgAAADEBAgANAAAAZgAAADIBAgAUAAAAdAAAAGmHBAABAAAAiAAAAJoAAABIAAAAAQAAAEgAAAABAAAAR0lNUCAyLjEwLjE0AAAyMDIxOjAzOjI1IDE2OjI5OjQ3AAEAAaADAAEAAAABAAAAAAAAAAgAAAEEAAEAAAAAAQAAAQEEAAEAAADXAAAAAgEDAAMAAAAAAQAAAwEDAAEAAAAGAAAABgEDAAEAAAAGAAAAFQEDAAEAAAADAAAAAQIEAAEAAAAGAQAAAgIEAAEAAAA7CQAAAAAAAAgACAAIAP/Y/+AAEEpGSUYAAQEAAAEAAQAA/9sAQwAIBgYHBgUIBwcHCQkICgwUDQwLCwwZEhMPFB0aHx4dGhwcICQuJyAiLCMcHCg3KSwwMTQ0NB8nOT04MjwuMzQy/9sAQwEJCQkMCwwYDQ0YMiEcITIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy/8AAEQgA1wEAAwEiAAIRAQMRAf/EAB8AAAEFAQEBAQEBAAAAAAAAAAABAgMEBQYHCAkKC//EALUQAAIBAwMCBAMFBQQEAAABfQECAwAEEQUSITFBBhNRYQcicRQygZGhCCNCscEVUtHwJDNicoIJChYXGBkaJSYnKCkqNDU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6g4SFhoeIiYqSk5SVlpeYmZqio6Slpqeoqaqys7S1tre4ubrCw8TFxsfIycrS09TV1tfY2drh4uPk5ebn6Onq8fLz9PX29/j5+v/EAB8BAAMBAQEBAQEBAQEAAAAAAAABAgMEBQYHCAkKC//EALURAAIBAgQEAwQHBQQEAAECdwABAgMRBAUhMQYSQVEHYXETIjKBCBRCkaGxwQkjM1LwFWJy0QoWJDThJfEXGBkaJicoKSo1Njc4OTpDREVGR0hJSlNUVVZXWFlaY2RlZmdoaWpzdHV2d3h5eoKDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uLj5OXm5+jp6vLz9PX29/j5+v/aAAwDAQACEQMRAD8A9/ooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAK+YP+GjvGH/QN0P8A78Tf/Ha+n6+AKAPYP+GjvGH/AEDdD/78Tf8Ax2j/AIaO8Yf9A3Q/+/E3/wAdrx+igD2D/ho7xh/0DdD/AO/E3/x2j/ho7xh/0DdD/wC/E3/x2vH6KAPYP+GjvGH/AEDdD/78Tf8Ax2j/AIaO8Yf9A3Q/+/E3/wAdrx+igD2D/ho7xh/0DdD/AO/E3/x2j/ho7xh/0DdD/wC/E3/x2vH6KAPYP+GjvGH/AEDdD/78Tf8Ax2j/AIaO8Yf9A3Q/+/E3/wAdrx+igD2D/ho7xh/0DdD/AO/E3/x2j/ho7xh/0DdD/wC/E3/x2vH6KAPYP+GjvGH/AEDdD/78Tf8Ax2j/AIaO8Yf9A3Q/+/E3/wAdrx+igD2D/ho7xh/0DdD/AO/E3/x2j/ho7xh/0DdD/wC/E3/x2vH6KAPYP+GjvGH/AEDdD/78Tf8Ax2j/AIaO8Yf9A3Q/+/E3/wAdrx+igD2D/ho7xh/0DdD/AO/E3/x2j/ho7xh/0DdD/wC/E3/x2vH6KAPYP+GjvGH/AEDdD/78Tf8Ax2j/AIaO8Yf9A3Q/+/E3/wAdrx+igD7/AKKKKACiiigAooooAKKKKACvgCvv+vgCgAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAPv8AooooAKKKKACiiigAooooAK+AK+/6+AKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooA+/wCiiigAooooAKKKKACiiigAr4Ar7/r4AoAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigD7/AKKKKACiiigAooooAKKKKACvgCvv+vgC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",
        }

        res = requests.post(url, json=req_body)
        resBody = res.json()

        self.assertEqual(200, res.status_code)
        self.assertEqual(768, resBody["dim"])


if __name__ == "__main__":
    unittest.main()
