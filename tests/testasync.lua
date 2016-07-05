require 'torch'
require 'nn'
require 'xlua'

-- Add to cpath
local info = debug.getinfo(1,'S')
local script_path=info.source:sub(2):gsub("testasync",'?')
package.path = package.path .. ";".. script_path .. ";../".. script_path..  ";../../" ..script_path
local modeltester = torch.TestSuite()
-- Inits the dataloaders
local audioload = paths.dofile('../init.lua')

local tester = torch.Tester()
ProFi = require 'ProFi'

local initcheck  = argcheck{
    {
        name='wavedatafile',
        type='string',
        help="Wavefile to run experiments with",
    },
    {
        name='htkdatafile',
        type='string',
        help='HTK filelist'
    }
}

local wavefilelist,htkfilelist = initcheck(arg[1],arg[2])


function modeltester:init()
    local filepath = wavefilelist
    local dataloader = audioload.WaveDataloader(filepath,100)
    local asyncdata = audioload.Asynciterator(dataloader)
end


function modeltester:testsampleiteratormultipleWAV()
    local filepath = wavefilelist
    local dataloader = audioload.WaveDataloader(filepath,400)
    local asyncdata = audioload.Asynciterator(dataloader,3)

    for s,e,inp,lab in dataloader:sampleiterator(128) do
        tester:assert(inp:size(1) == lab:size(1))
    end

    -- Multiple loops
    for s,e,inp,lab in asyncdata:sampleiterator(45) do
        tester:assert(inp:size(1) == lab:size(1))
    end

    for s,e,inp,lab in asyncdata:sampleiterator(45) do
        tester:assert(inp:size(1) == lab:size(1))
    end
end

function modeltester:testsampleiteratormultipleHTK()
    local filepath = htkfilelist
    local dataloader = audioload.HtkDataloader(filepath)
    local asyncdata = audioload.Asynciterator(dataloader,5)

    for s,e,inp,lab in dataloader:sampleiterator(128) do
        tester:assert(inp:size(1) == lab:size(1))
    end

    -- Multiple loops
    for s,e,inp,lab in asyncdata:sampleiterator(45) do
        tester:assert(inp:size(1) == lab:size(1))
    end

    for s,e,inp,lab in asyncdata:sampleiterator(45) do
        tester:assert(inp:size(1) == lab:size(1))
    end
end

function modeltester:testsampleiteratorsinglethread()
    local filepath = wavefilelist
    local dataloader = audioload.WaveDataloader(filepath,100)
    local asyncdata = audioload.Asynciterator(dataloader,1)

    for s,e,inp,lab in asyncdata:sampleiterator(45) do
        tester:assert(inp:size(1) == lab:size(1))
    end
end

tester:add(modeltester)
tester:run()
