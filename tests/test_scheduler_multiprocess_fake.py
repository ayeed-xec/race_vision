from human_vision.concurrency.multiprocess import MultiprocessScheduler


def test_multiprocess_scheduler_runs_tasks():
    scheduler = MultiprocessScheduler(max_workers=2)
    tasks = {
        "a": lambda: 1,
        "b": lambda: 2,
    }
    results = scheduler.run(tasks)
    scheduler.close()
    assert results == {"a": 1, "b": 2}
