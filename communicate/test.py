from game import test_game
import global_var as gl
from game import test_game
import threading

def ll():
    while True:
        output = gl.get_value("output")
        print(output)

if __name__=="__main__":
    gl._init()
    thread = threading.Thread(target=ll)
    thread.start()
    test_game.test_main()
