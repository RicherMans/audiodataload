require 'torch'
require 'nn'
require 'xlua'
require 'sys'

-- Add to cpath
local info = debug.getinfo(1,'S')
local script_path=info.source:sub(2):gsub("testhdf5iterator",'?')
package.path = package.path .. ";".. script_path .. ";../".. script_path..  ";../../" ..script_path
local modeltester = torch.TestSuite()
-- Inits the dataloaders
local audioload = paths.dofile('../init.lua')

local tester = torch.Tester()

ProFi = require 'ProFi'

function modeltester:init()
    local filepath = "train.lst"
    local dataloader = audioload.WaveDataloader(filepath,100)

    local wrapper = audioload.Hdf5iterator{module=dataloader,filepath='myfile',chunksize=10000}
    sys.fexecute("rm myfile")
end

function modeltester:iterwavedataloaderrandom()
    local filepath = "train.lst"
    local dataloader = audioload.WaveDataloader{path=filepath,framesize=100}

    local wrapper = audioload.Hdf5iterator{module=dataloader,filepath='myfile2'}
    ProFi:start('hdf5')
    local it = wrapper:sampleiterator(128,nil,true)
    local tic = torch.tic()
    for i,k,v,d in it do
        -- print(i,k,v,d)
    end
    ProFi:stop()
    it = dataloader:sampleiterator(128,nil,true)
    tic = torch.tic()
    for i,k,v,d in it do

    end
    ProFi:writeReport('testwavedataloaderrandomhdf5.txt')
    sys.fexecute("rm myfile2")
end

function modeltester:iterwavedataloader()
    local filepath = "train.lst"
    local dataloader = audioload.WaveDataloader{path=filepath,framesize=100}

    local wrapper = audioload.Hdf5iterator{module=dataloader,filepath='myfile2'}

    local it = wrapper:sampleiterator(128,nil,false)
    local tic = torch.tic()
    for i,k,v,d in it do
        -- print(i,k,v,d)
    end
    print("\nHDF took " .. torch.toc(tic))

    it = dataloader:sampleiterator(128,nil,false)
    tic = torch.tic()
    for i,k,v,d in it do

    end
    print("\nWave took " .. torch.toc(tic))
    sys.fexecute("rm myfile2")
end
--
function modeltester:iterseqence()
    local filepath = "train.lst"
    local dataloader = audioload.WaveDataloader(filepath,100)

    local seqdataloader = audioload.Sequenceiterator(dataloader,20)

    local wrapper = audioload.Hdf5iterator{module=seqdataloader,filepath='myfile2'}

    local it = wrapper:sampleiterator(128,nil,true)
    local tic = torch.tic()
    for i,k,v,d in it do
        -- print(v:size())
        -- print(i,k,v,d)
    end
    print("\nHDF Took "..torch.toc(tic))

    sys.fexecute("rm myfile2")
end
--

function modeltester:iterutt()
    local filepath = "train.lst"
    local dataloader = audioload.WaveDataloader(filepath,100)

    local wrapper = audioload.Hdf5iterator{module=dataloader,filepath='myfile1'}

    local it = wrapper:uttiterator(1)
    local tic = torch.tic()
    for i,k,v,d in it do
        -- xlua.progress(i,k)
        -- print(i,k,v,d)
    end
    print("Took "..torch.toc(tic))
    sys.fexecute("rm myfile1")
end


tester:add(modeltester)
tester:run()
