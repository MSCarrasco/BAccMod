# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: toolbox.py
# Purpose: Definition of the Exception specific to the BAccMod code
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# ---------------------------------------------------------------------


class BackgroundModelFormatException(Exception):
    def __init__(self, *args):
        super().__init__(*args)
