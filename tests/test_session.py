import time

from dataset.session import DatasetSession


session = DatasetSession(
    save_interval=2.0
)

session.start()

while True:

    if session.should_save():

        print(
            "Save!",
            session.saved_count()
        )

    time.sleep(0.1)