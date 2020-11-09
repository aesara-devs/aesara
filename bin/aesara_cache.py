#!/usr/bin/env python

import logging
import os
import sys

if sys.platform == 'win32':
    config_for_aesara_cache_script = 'cxx=,device=cpu'
    aesara_flags = os.environ['THEANO_FLAGS'] if 'THEANO_FLAGS' in os.environ else ''
    if aesara_flags:
        aesara_flags += ','
    aesara_flags += config_for_aesara_cache_script
    os.environ['THEANO_FLAGS'] = aesara_flags

import aesara
from aesara import config
import aesara.gof.compiledir
from aesara.gof.cc import get_module_cache

_logger = logging.getLogger('aesara.bin.aesara-cache')


def print_help(exit_status):
    if exit_status:
        print('command "%s" not recognized' % (' '.join(sys.argv)))
    print('Type "aesara-cache" to print the cache location')
    print('Type "aesara-cache help" to print this help')
    print('Type "aesara-cache clear" to erase the cache')
    print('Type "aesara-cache list" to print the cache content')
    print('Type "aesara-cache unlock" to unlock the cache directory')
    print('Type "aesara-cache cleanup" to delete keys in the old '
          'format/code version')
    print('Type "aesara-cache purge" to force deletion of the cache directory')
    print('Type "aesara-cache basecompiledir" '
          'to print the parent of the cache directory')
    print('Type "aesara-cache basecompiledir list" '
          'to print the content of the base compile dir')
    print('Type "aesara-cache basecompiledir purge" '
          'to remove everything in the base compile dir, '
          'that is, erase ALL cache directories')
    sys.exit(exit_status)


def main():
    if len(sys.argv) == 1:
        print(config.compiledir)
    elif len(sys.argv) == 2:
        if sys.argv[1] == 'help':
            print_help(exit_status=0)
        if sys.argv[1] == 'clear':
            # We skip the refresh on module cache creation because the refresh will
            # be done when calling clear afterwards.
            cache = get_module_cache(init_args=dict(do_refresh=False))
            cache.clear(unversioned_min_age=-1, clear_base_files=True,
                        delete_if_problem=True)

            # Print a warning if some cached modules were not removed, so that the
            # user knows he should manually delete them, or call
            # aesara-cache purge, # to properly clear the cache.
            items = [item for item in sorted(os.listdir(cache.dirname))
                     if item.startswith('tmp')]
            if items:
                _logger.warning(
                    'There remain elements in the cache dir that you may '
                    'need to erase manually. The cache dir is:\n  %s\n'
                    'You can also call "aesara-cache purge" to '
                    'remove everything from that directory.' %
                    config.compiledir)
                _logger.debug('Remaining elements (%s): %s' %
                              (len(items), ', '.join(items)))
        elif sys.argv[1] == 'list':
            aesara.gof.compiledir.print_compiledir_content()
        elif sys.argv[1] == 'cleanup':
            aesara.gof.compiledir.cleanup()
            cache = get_module_cache(init_args=dict(do_refresh=False))
            cache.clear_old()
        elif sys.argv[1] == 'unlock':
            aesara.gof.compilelock.force_unlock()
            print('Lock successfully removed!')
        elif sys.argv[1] == 'purge':
            aesara.gof.compiledir.compiledir_purge()
        elif sys.argv[1] == 'basecompiledir':
            # Simply print the base_compiledir
            print(aesara.config.base_compiledir)
        else:
            print_help(exit_status=1)
    elif len(sys.argv) == 3 and sys.argv[1] == 'basecompiledir':
        if sys.argv[2] == 'list':
            aesara.gof.compiledir.basecompiledir_ls()
        elif sys.argv[2] == 'purge':
            aesara.gof.compiledir.basecompiledir_purge()
        else:
            print_help(exit_status=1)
    else:
        print_help(exit_status=1)


if __name__ == '__main__':
    main()
