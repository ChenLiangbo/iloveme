#!/user/bin/env/python
#coding:utf-8

# docs http://docs.pythontab.com/tornado/introduction-to-tornado/ch1.html#ch1-1-1
# http://www.tornadoweb.org/en/stable/web.html
import json
import os.path
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
from tornado.options import define, options
import URLhandler

define("port", default=5656, help="run on the given port", type=int)

app = tornado.web.Application(
    handlers=[(r'/login/', URLhandler.LoginHandler),
              (r'/stations/', URLhandler.StationsHandler),
              (r'/alert/',URLhandler.AlertHandler),
              (r'/device/',URLhandler.DeviceHandler),
              (r'/chart/',URLhandler.ChartHandler),
              (r'/alert_search/',URLhandler.AlertSearchHandler),
              (r'/position/',URLhandler.UpdatePositionHandler),
              (r'/feedback/',URLhandler.FeedbackHandler)],           
    debug = True,
    cookie_secret = '7cbddfc12c7522bc46010a4563e80257',
    # template_path = os.path.join(os.path.dirname(__file__), "helloword2"),
    # static_path=os.path.join(os.path.dirname(__file__), "helloword2"),
     
)

def appStart():
    tornado.options.parse_command_line()
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()

if __name__ == "__main__":
	appStart()
