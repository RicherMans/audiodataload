require 'torch'
require 'nn'
require 'xlua'

-- Add to cpath
local info = debug.getinfo(1,'S')
local script_path=info.source:sub(2):gsub("testseqiterator",'?')
package.path = package.path .. ";".. script_path .. ";../".. script_path..  ";../../" ..script_path
local modeltester = torch.TestSuite()
-- Inits the dataloaders
local audioload = paths.dofile('../init.lua')

local tester = torch.Tester()


function modeltester:init()
    local filepath = "train.lst"
    local dataloader = audioload.WaveDataloader(filepath,100)
    audioload.Sequenceiterator(dataloader,5)
end

function modeltester:sampleiter()
    local filepath = "train.lst"
    local seqlenth = 50
    local framesize = 100
    local dataloader = audioload.WaveDataloader(filepath,framesize)
    local seqiter = audioload.Sequenceiterator(dataloader,seqlenth)
    for start,all,input,target in seqiter:sampleiterator(128,nil,true)do
        tester:asserteq(input:dim(),3)
        tester:asserteq(input:size(1),seqlenth)
        tester:asserteq(input:size(3),framesize)
        tester:asserteq(input:size(2),target:size(1))
    end
end

function modeltester:sampleiterlarge()
    local filepath = "train.lst"
    local framesize = 100
    local dataloader = audioload.WaveDataloader(filepath,framesize)
    local seqiter = audioload.Sequenceiterator{wrappedmodule=dataloader,usemaxseqlength =true}
    for start,all,input,target in seqiter:sampleiterator(128,nil,true)do
        tester:asserteq(input:dim(),3)
        -- tester:asserteq(input:size(1),seqlenth)
        tester:asserteq(input:size(3),framesize)
        tester:asserteq(input:size(2),target:size(1))
    end
end

function modeltester:uttiter()
    local filepath = "train.lst"
    local seqlenth = 50
    local framesize = 100
    local dataloader = audioload.WaveDataloader(filepath,framesize)
    local seqiter = audioload.Sequenceiterator(dataloader,seqlenth)
    local tic = torch.tic()
    for start,all,input,target,fpath in seqiter:uttiterator(128,nil,true)do
        tester:asserteq(input:dim(),3)
        tester:asserteq(input:size(1),seqlenth)
        tester:asserteq(input:size(3),framesize)
        tester:asserteq(input:size(2),target:size(1))
    end
    print("\n"..torch.toc(tic))
end

function modeltester:uttiterlarge()
    local filepath = "train.lst"
    local framesize = 500
    local dataloader = audioload.WaveDataloader(filepath,framesize)
    local seqiter = audioload.Sequenceiterator(dataloader,1,true)
    local tic = torch.tic()
    for start,all,input,target in seqiter:uttiterator(128,nil)do
        tester:asserteq(input:dim(),3)
        tester:asserteq(input:size(3),framesize)
        tester:asserteq(input:size(2),target:size(1))
    end
    print("\n"..torch.toc(tic))
end
function modeltester:uttiterfull()
    local filepath = "train.lst"
    local framesize = 200
    local dataloader = audioload.WaveDataloader(filepath,framesize)
    local seqiter = audioload.Sequenceiterator{wrappedmodule=dataloader,usemaxseqlength=true}
    local tic = torch.tic()
    for start,all,input,target in seqiter:uttiterator(128,nil,true)do
        tester:asserteq(input:dim(),3)
        tester:asserteq(input:size(3),framesize)
        tester:asserteq(input:size(2),target:size(1))
    end
    print("\n"..torch.toc(tic))
end


tester:add(modeltester)
tester:run()
