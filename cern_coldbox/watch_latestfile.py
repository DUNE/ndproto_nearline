#!/usr/bin/env python3
# Check input_folder of a workflow-runner config every --interval and run the
# specified workflow on each new file once it has stopped growing.
# Files already there at startup are skipped (use `workflow-runner batch` for those),
# unless they are still being written.
import argparse
import logging
import os
import time
from pathlib import Path

from workflow_runner.core import WorkflowRunner

logger = logging.getLogger('watch_latestfile')


def parse_interval(text):
    '''"60", "60s", "1m", "2h" -> seconds'''
    units = {'s': 1, 'm': 60, 'h': 3600}
    if text[-1] in units:
        return float(text[:-1]) * units[text[-1]]
    return float(text)


def list_files(config):
    files = {}
    for pattern in config.file_patterns:
        glob_pattern = '**/' + pattern if config.recursive else pattern
        for path in config.input_folder.glob(glob_pattern):
            if path.is_file():
                files[path.resolve()] = path.stat()
    return files


def main(config, interval):
    # the workflow commands find the scripts through this variable
    os.environ.setdefault('COLDBOX_NEARLINE_DIR', str(Path(__file__).resolve().parent))
    runner = WorkflowRunner(config)
    interval = parse_interval(interval)

    last = {path: st.st_size for path, st in list_files(runner.config).items()}
    done = set(last)
    logger.info('Checking {} every {:g} s ({} files already there are skipped)'.format(
        runner.config.input_folder, interval, len(done)))

    while True:
        time.sleep(interval)
        files = list_files(runner.config)
        for path, st in sorted(files.items(), key=lambda item: item[1].st_mtime):
            if path in done and st.st_size != last.get(path):
                done.discard(path)  # still being written
            stable = (st.st_size == last.get(path) and st.st_size > 0 and
                      time.time() - st.st_mtime >= runner.config.stability_check_time)
            if path not in done and stable:
                logger.info('New file: {}'.format(path.name))
                runner.process_file(path)
                done.add(path)
        last = {path: st.st_size for path, st in files.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='''workflow-runner config yaml''')
    parser.add_argument('--interval', default='1m', type=str,
                        help='''How often to check for new files, e.g. 30s, 1m, 5m (default: %(default)s)''')
    args = parser.parse_args()
    try:
        main(**vars(args))
    except KeyboardInterrupt:
        logger.info('Stopped')
