import sys
from bionemo.evo2.utils.logging.bnm_call_stack_monitor import BnmCallStackMonitor


# Example usage
def foo(x, y):
    return bar(x) + y


def bar(z):
    return z * 2

def main():

    monitor = BnmCallStackMonitor()
    monitor.start_monitoring()

    result = foo(3, 4)

    monitor.stop_monitoring()

    monitor.write_events_to_file()


if __name__ == "__main__":
    main()