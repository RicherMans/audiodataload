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
    audioload.Sequenceiterator(dataloader,5)
end

-- function modeltester:sampleiter()
--     local filepath = wavefilelist
--     local seqlenth = 50
--     local framesize = 100
--     local dataloader = audioload.WaveDataloader(filepath,framesize)
--     local seqiter = audioload.Sequenceiterator(dataloader,seqlenth)
--     for start,all,input,target in seqiter:sampleiterator(128,nil,true)do
--         tester:asserteq(input:dim(),3)
--         tester:asserteq(input:size(1),seqlenth)
--         tester:asserteq(input:size(3),framesize)
--         tester:asserteq(input:size(2),target:size(1))
--     end
-- end

function modeltester:sampleiterlarge()
    local filepath = wavefilelist
    local framesize = 100
    local dataloader = audioload.WaveDataloader(filepath,framesize)
    local seqiter = audioload.Sequenceiterator{wrappedmodule=dataloader}
    for start,all,input,target in seqiter:sampleiterator(128,nil)do
        tester:asserteq(input:dim(),3)
        -- tester:asserteq(input:size(1),seqlenth)
        tester:asserteq(input:size(3),framesize)
        tester:asserteq(input:size(2),target:size(1))
    end
end

function modeltester:uttiter()
    local filepath = wavefilelist
    local seqlenth = 50
    local framesize = 100
    local dataloader = audioload.WaveDataloader(filepath,framesize)
    local seqiter = audioload.Sequenceiterator(dataloader,seqlenth)
    local tic = torch.tic()
    for start,all,input,target,fpath in seqiter:uttiterator(128,nil)do
        tester:asserteq(input:dim(),3)
        tester:asserteq(input:size(1),seqlenth)
        tester:asserteq(input:size(3),framesize)
        tester:asserteq(input:size(2),target:size(1))
    end
    print("\n"..torch.toc(tic))
end

-- function modeltester:uttiterlarge()
--     local filepath = wavefilelist
--     local framesize = 500
--     local dataloader = audioload.WaveDataloader(filepath,framesize)
--     local seqiter = audioload.Sequenceiterator(dataloader,1,true)
--     local tic = torch.tic()
--     for start,all,input,target in seqiter:uttiterator(128,nil)do
--         tester:asserteq(input:dim(),3)
--         tester:asserteq(input:size(3),framesize)
--         tester:asserteq(input:size(2),target:size(1))
--     end
--     print("\n"..torch.toc(tic))
-- end
-- function modeltester:uttiterfull()
--     local filepath = wavefilelist
--     local framesize = 200
--     local dataloader = audioload.WaveDataloader(filepath,framesize)
--     local seqiter = audioload.Sequenceiterator{wrappedmodule=dataloader,usemaxseqlength=true}
--     local tic = torch.tic()
--     for start,all,input,target in seqiter:uttiterator(128,nil,true)do
--         tester:asserteq(input:dim(),3)
--         tester:asserteq(input:size(3),framesize)
--         tester:asserteq(input:size(2),target:size(1))
--     end
--     print("\n"..torch.toc(tic))
-- end

function modeltester:iterutthtk()
    local dataloader = audioload.HtkDataloader(htkfilelist)
    local seqiter = audioload.Sequenceiterator{wrappedmodule=dataloader,usemaxseqlength=true}
    for start,all,input,target in seqiter:uttiterator(5) do
        
    end 
end

function modeltester:itersamphtk()
    local dataloader = audioload.HtkDataloader(htkfilelist)
    local seqiter = audioload.Sequenceiterator{wrappedmodule=dataloader,seqlen=3}
    for start,all,input,target in seqiter:sampleiterator(128) do
    end 
end

function modeltester:itersamphtklarge()
    local dataloader = audioload.HtkDataloader(htkfilelist)
    local seqiter = audioload.Sequenceiterator{wrappedmodule=dataloader,seqlen=1000}
    for start,all,input,target in seqiter:sampleiterator(128) do
    end 
end

tester:add(modeltester)
tester:run()
