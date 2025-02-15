"""Common gui utilities"""

"""___Built-In Modules___"""
from typing import Callable

"""___Third-Party Modules___"""
from qtpy import QtCore

"""___LIME_TBX Modules___"""
from lime_tbx.datatypes import logger


class WorkerStopper:
    def __init__(self):
        self.mutex = QtCore.QMutex()
        self.running = True


class CallbackWorker(QtCore.QObject):
    finished = QtCore.Signal(list)
    exception = QtCore.Signal(Exception)
    info = QtCore.Signal(str)

    def __init__(
        self,
        callback: Callable,
        args: list,
        with_info: bool = False,
        stoppable: bool = False,
    ):
        super().__init__()
        self.callback = callback
        self.args = args
        self.with_info = with_info
        self.stoppable = stoppable
        if self.stoppable:
            self.stopper = WorkerStopper()

    def run(self):
        try:
            if not self.with_info:
                if not self.stoppable:
                    res = self.callback(*self.args)
                else:
                    res = self.callback(*self.args, self.stopper)
            else:
                if not self.stoppable:
                    res = self.callback(*self.args, self.info)
                else:
                    res = self.callback(*self.args, self.info, self.stopper)
            if res is not None:
                self.finished.emit(list(res))
            else:
                self.finished.emit([])
        except Exception as e:
            logger.get_logger().exception(e)
            self.exception.emit(e)

    @QtCore.Slot()
    def stop(self):
        if self.stoppable:
            self.stopper.mutex.lock()
            self.stopper.running = False
            self.stopper.mutex.unlock()
        else:
            raise Exception("Tried to stop a nonstoppable CallbackWorker.")


def start_thread(
    worker: CallbackWorker,
    worker_th: QtCore.QThread,
    finished: Callable,
    error: Callable,
    info: Callable = None,
):
    worker.moveToThread(worker_th)
    worker_th.started.connect(worker.run)
    worker.finished.connect(worker_th.quit)
    worker.finished.connect(worker.deleteLater)
    if finished:
        worker.finished.connect(finished)
    worker.exception.connect(worker_th.quit)
    worker.exception.connect(worker.deleteLater)
    if error:
        worker.exception.connect(error)
    if info:
        worker.info.connect(info)
    worker_th.finished.connect(worker_th.deleteLater)
    worker_th.start()
