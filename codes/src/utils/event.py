from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path


def get_log_probs(event_path):
    event_acc = EventAccumulator(str(event_path))
    event_acc.Reload()
    # print(event_acc.Tags()) # Show all tags in the log file

    _, step_nums, training_log_probs = zip(*event_acc.Scalars("training_log_probs"))
    _, _, validation_log_probs = zip(*event_acc.Scalars("validation_log_probs"))
    _, _, best_validation_log_prob = zip(*event_acc.Scalars("best_validation_log_prob"))
    return step_nums, training_log_probs, validation_log_probs, best_validation_log_prob


def get_event_values(scalar_events):
    # Extract the required fields from the scalar events
    wall_time, step_nums, values = [], [], []
    for event in scalar_events:
        wall_time.append(event.wall_time)
        step_nums.append(event.step)
        values.append(event.value)

    return wall_time, step_nums, values


def get_train_valid_lr(log_dir, use_loss=False):
    log_dir = Path(log_dir)

    lr_path = sorted(log_dir.glob("*events.out.tfevents.*"))[-1]
    if not use_loss:
        train_path = sorted(log_dir.glob("*log_probs_training/events.out.tfevents.*"))[-1]
        valid_path = sorted(log_dir.glob("*log_probs_validation/events.out.tfevents.*"))[-1]
    else:
        train_path = sorted(log_dir.glob("*loss_training/events.out.tfevents.*"))[-1]
        valid_path = sorted(log_dir.glob("*loss_validation/events.out.tfevents.*"))[-1]

    # load learning rate
    lr_event = EventAccumulator(str(lr_path))
    lr_event.Reload()
    print(lr_event.Tags())
    scalar_events = lr_event.Scalars("learning_rates")
    wall_time, step_nums, learning_rates = get_event_values(scalar_events)

    # load training log probs
    train_event = EventAccumulator(str(train_path))
    train_event.Reload()
    print(train_event.Tags())
    if not use_loss:
        scalar_events = train_event.Scalars("log_probs")
    else:
        scalar_events = train_event.Scalars("loss")
    _, _, log_probs_train = get_event_values(scalar_events)

    # load validation log probs
    valid_event = EventAccumulator(str(valid_path))
    valid_event.Reload()
    print(valid_event.Tags())
    if not use_loss:
        scalar_events = valid_event.Scalars("log_probs")
    else:
        scalar_events = valid_event.Scalars("loss")
    _, _, log_probs_valid = get_event_values(scalar_events)

    return wall_time, step_nums, learning_rates, log_probs_train, log_probs_valid
