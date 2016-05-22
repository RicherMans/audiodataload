require 'paths'
require 'xlua'
require 'torchx'
require 'string'
require 'os'
require 'sys'
ffi = require 'ffi'
argcheck = require 'argcheck'

-- make unpack visible in global
unpack = unpack or table.unpack

floor = math.floor

local adl = require '_base'

require 'BaseDataloader'
require 'WaveDataloader'
require 'Hdf5iterator'


return adl
