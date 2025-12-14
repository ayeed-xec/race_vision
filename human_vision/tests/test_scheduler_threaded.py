from human_vision.concurrency.threaded import ThreadedScheduler


def test_threaded_scheduler_runs_tasks():
    scheduler = ThreadedScheduler(max_workers=2)
    tasks = {
        "a": lambda: 1,
        "b": lambda: 2,
    }
    results = scheduler.run(tasks)
    scheduler.close()
    assert results == {"a": 1, "b": 2}
