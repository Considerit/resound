import os
import sys
from http.server import SimpleHTTPRequestHandler, HTTPServer
from urllib.parse import unquote

class CustomHandler(SimpleHTTPRequestHandler):
    BLACKLIST = ['.env']  # Add the files you want to blacklist

    def do_GET(self):
        if self.is_blacklisted(self.path):
            self.send_error(403, "Forbidden")
        else:
            print(self.path.strip('/'), self.BLACKLIST)
            super().do_GET()

    def list_directory(self, path):
        try:
            list = os.listdir(path)
        except os.error:
            self.send_error(404, "No permission to list directory")
            return None
        list.sort(key=lambda a: a.lower())
        f = self.wfile
        displaypath = unquote(self.path)

        try:
            f.write('HTTP/1.1 200 OK\r\n'.encode())
            f.write('Content-Type: text/html; charset=utf-8\r\n'.encode())
            f.write('\r\n'.encode())

            f.write(f'<html>\n<head>\n<title>Directory listing for {displaypath}</title>\n</head>\n<body>\n'.encode())
            f.write(f'<h2>Directory listing for {displaypath}</h2>\n<hr>\n<ul>\n'.encode())
            for name in list:
                if name in self.BLACKLIST:
                    continue
                fullname = os.path.join(path, name)
                displayname = name + ("/" if os.path.isdir(fullname) else "")
                f.write(f'<li><a href="{displaypath}{name}">{displayname}</a></li>\n'.encode())
            f.write('</ul>\n<hr>\n</body>\n</html>\n'.encode())
        except BrokenPipeError as e:
            # Log the error or pass if logging is not set up
            pass
        except Exception as e:
            # Handle other exceptions
            pass

        return None


    def is_blacklisted(self, path):
        # Extract the file name from the path and check if it's in the blacklist
        filename = os.path.basename(unquote(path))
        return filename in self.BLACKLIST

    def guess_type(self, path):
        if path.endswith(('.js', '.ts', '.html', '.css', '.json', '.log', '.py', '.txt', '.jsx', '.tsx')):
            return 'text/plain; charset=utf-8'
        return SimpleHTTPRequestHandler.guess_type(self, path)

def run(server_class=HTTPServer, handler_class=CustomHandler, directory=None, port=8000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    if directory:
        os.chdir(directory)
    print(f'Server running at http://localhost:{port}/ serving {directory or os.getcwd()}')
    httpd.serve_forever()

if __name__ == '__main__':
    directory = sys.argv[1] if len(sys.argv) > 1 else None
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    run(directory=directory, port=port)
