require 'paths'
require 'xlua'
require 'string'
require 'os'
require 'sys'
ffi = require 'ffi'
require 'torchx'
argcheck = require 'argcheck'

-- make unpack visible in global
unpack = unpack or table.unpack

floor = math.floor

local adl = require 'audiodataload._base'

require 'audiodataload.BaseDataloader'
require 'audiodataload.WaveDataloader'
require 'audiodataload.Hdf5iterator'
require 'audiodataload.Sequenceiterator'
require 'audiodataload.HtkDataloader'
require 'audiodataload.Asynciterator'

return adl
