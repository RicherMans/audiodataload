require 'torch'
require 'nn'
require 'xlua'

-- Add to cpath
local info = debug.getinfo(1,'S')
local script_path=info.source:sub(2):gsub("testhtkdataloader",'?')
package.path = package.path .. ";".. script_path .. ";../".. script_path..  ";../../" ..script_path
local modeltester = torch.TestSuite()
-- Inits the dataloaders
local audioload = paths.dofile('../init.lua')

local tester = torch.Tester()

local filelist = arg[1]

function modeltester:init()
    local filepath = filelist
    local dataloader = audioload.HtkDataloader(filepath)
end
function modeltester:testbatchUtterance()
    local filepath = filelist
    local dataloader = audioload.HtkDataloader{path=filepath}

    local iter = dataloader:uttiterator()
    for i,k,v,t in iter do
        tester:assert(type(i) == 'number')
        tester:assert(type(k) == 'number')
        tester:assert(torch.isTensor(v))
    end
end
function modeltester:testnonrandomizedsamples()
    local filepath = filelist
    local dataloader = audioload.HtkDataloader(filepath)
    local classsizes= torch.Tensor(dataloader:nClasses()):zero()

    local batchsize = 128
    for s,e,k,v in dataloader:sampleiterator(batchsize) do
        local addone = torch.Tensor(v:size(1)):fill(1)
        classsizes:indexAdd(1,v:long(),addone)
    end
    tester:assert(classsizes:sum()==dataloader:size())
    for i=1,3 do
        local tmpclasssizes = torch.Tensor(dataloader:nClasses()):zero()
        for s,e,k,v in dataloader:sampleiterator(batchsize) do
            local addone = torch.Tensor(v:size(1)):fill(1)
            tmpclasssizes:indexAdd(1,v:long(),addone)
        end
        tester:eq(tmpclasssizes,classsizes)
    end

end

--
function modeltester:testrandomize()
    local filepath = filelist
    local dataloader = audioload.HtkDataloader{path=filepath}

    local valuetolab = {}
    local numvalues = 0
    local tic = torch.tic()
    local labcount = torch.zeros(dataloader:nClasses())
    for s,e,inp,lab in dataloader:sampleiterator(1,nil,true) do
        valuetolab[inp[1][1]] = lab[1]
        numvalues = numvalues + 1
        labcount[lab[1]] = labcount[lab[1]] + 1
        tester:assert(inp:size(1) == lab:size(1))
    end

    tester:assert(labcount:sum() == dataloader:size())

    -- Emulate some 5 iterations over the data
    for i=1,5 do
        local tmpnumvalues = numvalues
        local tmplabcount = labcount:clone()
        for s,e,inp,lab in dataloader:sampleiterator(1,nil,true) do
            if valuetolab[inp[1][1]] then
                if valuetolab[inp[1][1]] == lab[1] then
                    tmpnumvalues = tmpnumvalues - 1
                end
            end
            tmplabcount[lab[1]] = tmplabcount[lab[1]] -1
            tester:assert(inp:size(1) == lab:size(1))
        end
        tester:eq(tmplabcount,torch.zeros(dataloader:nClasses()),"Labels are not the same in iteration "..i)
        tester:assert(tmpnumvalues == 0,"Error in iteration "..i.. " difference is "..tmpnumvalues)
    end
    
end

function modeltester:testsize()
    local dataloader = audioload.HtkDataloader{path=filelist}
    local size = dataloader:size()
    -- Simulate 3 iterations over the dataset
    for i=1,3 do
        local numsamples = 0
        for _ in dataloader:sampleiterator(1,nil,true) do
            numsamples = numsamples + 1
        end
        tester:assert(size == numsamples)
    end
end

function modeltester:testloadAudioSample()
    local dataloader = audioload.HtkDataloader(filelist)
    local f=assert(io.open(filelist))
    local firstline = f:read()
    local feat = firstline:split(" ")[1]
    -- print(dataloader:loadAudioSample(feat,39,40))
    -- dataloader:loadAudioSample
end

function modeltester:testusize()
    local dataloader = audioload.HtkDataloader{path=filelist}
    local size = dataloader:usize()
    -- Simulate 3 iterations over the dataset
    for i=1,3 do
        local numutts = 0
        for _ in dataloader:uttiterator() do
            numutts = numutts + 1
        end
        tester:assert(size == numutts)
    end
end
--
-- function modeltester:testUtteranceSeq()
--     local filepath = "train.lst"
--     local dataloader = audioload.HtkDataloader{path=filepath}
--
--     local it = dataloader:uttiterator(1)
--     local tic = torch.tic()
--     for i,k,v,t in it do
--         tester:assert(i ~= nil)
--         tester:assert(k ~= nil)
--         tester:assert(v:nonzero() ~= nil)
--         tester:assert(v ~= nil)
--         tester:assert(t ~= nil)
--         -- xlua.progress(i,k)
--     end
--     print("Took "..torch.toc(tic))
-- end
--
-- function modeltester:testUtteranceNoSeq()
--     local filepath = "train.lst"
--     local dataloader = audioload.HtkDataloader{path=filepath}
--
--     local tic = torch.tic()
--     local it = dataloader:uttiterator(1)
--     for i,k,v,t in it do
--         tester:assert(i ~= nil)
--         tester:assert(k ~= nil)
--         tester:assert(v:nonzero() ~= nil)
--         tester:assert(v ~= nil)
--         tester:assert(t ~= nil)
--         -- xlua.progress(i,k)
--         -- print(i,k,v,d)
--     end
--     print("Took "..torch.toc(tic))
-- end


if not filelist or filelist == "" then
    print("Please pass a filelist as first argument")
    return
end

tester:add(modeltester)
tester:run()
