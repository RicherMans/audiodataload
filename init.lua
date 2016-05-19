require 'paths'
require 'xlua'
require 'torchx'
require 'string'
require 'os'
require 'sys'
require 'ffi'
argcheck = require 'argcheck'

-- make unpack visible in global
unpack = unpack or table.unpack

local adl = require '_base'

require 'BaseDataloader'
require 'WaveDataloader'
require 'Hdf5iterator'


return adl
