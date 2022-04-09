#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:        
#   Author:         xiahong xiahahaha01@gmail.com
#   Create:         

"""one line of summary

overall description of the module or program.

Typical usage example:
foo = ClassFoo()
bar = foo.FunctionBar()
"""
#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       {{FILE}}
#   Author:         {{NAME}} {{EMAIL}}
#   Create:         {{TODAY}}
#   Description:    ---
"""one line of summary

overall description of the module or program.

Typical usage example:
foo = ClassFoo()
bar = foo.FunctionBar()
"""
line1
line2
line3#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       ss
#   Author:         xiahong xiahahaha01@gmail.com
#   Create:         07/04/2022
#   Description:    ---
"""one line of summary
123
overall description of the module or program.

Typical usage example:
foo = ClassFoo()
bar = foo.FunctionBar()
"""

import os
import sys

import argparse

def main():
    """one line of summary.

    overall description.

    Args:
        ...

    Returns:
        ...

    Raise:
        ...

    """
    return None

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--config_path", default="./config.py")
    args = parser.parse_args()
    main(args.config_path)

