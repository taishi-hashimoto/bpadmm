def test_cosine_decay_schedule():
    from bpadmm import cosine_decay_schedule

    # Single value test
    print(cosine_decay_schedule(10, 1, 0.1))
    # Batch of values test
    print(cosine_decay_schedule(10, [2, 1], [0.3, 0.1]))
