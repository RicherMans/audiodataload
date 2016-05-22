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


function modeltester:init()
    local filepath = "train.lst"
    local dataloader = audioload.WaveDataloader(filepath,100)

    local wrapper = audioload.Hdf5iterator{module=dataloader,newfile='myfile',chunksize=10000}
    sys.fexecute("rm myfile")
end
--
function modeltester:itersample()
    local filepath = "train.lst"
    local dataloader = audioload.WaveDataloader(filepath,100)

    local wrapper = audioload.Hdf5iterator{module=dataloader,newfile='myfile2',chunksize=50000}
    wrapper:random()

    local it = wrapper:sampleiterator(100)
    local tic = torch.tic()
    for i,k,v,d in it do
        -- xlua.progress(i,k)
        -- print(i,k,v,d)
    end
    print("Took "..torch.toc(tic))


    tic = torch.tic()
    it = dataloader:sampleiterator(100)
    for i,k,v,d in it do
        -- xlua.progress(i,k)
        -- print(i,k,v,d)
    end
    print("Took "..torch.toc(tic))

    sys.fexecute("rm myfile2")
end


function modeltester:iterutt()
    local filepath = "train.lst"
    local dataloader = audioload.WaveDataloader(filepath,100)

    local wrapper = audioload.Hdf5iterator{module=dataloader,newfile='myfile1'}
    wrapper:random()

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
