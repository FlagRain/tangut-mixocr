"""python -m app：按 config 里的地址和端口启动网站（单进程；识别在后台线程里排队进行）。"""
import uvicorn

from . import config as C

uvicorn.run("app.main:app", host=C.HOST, port=C.PORT, workers=1, proxy_headers=False, access_log=False)
