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

    local wrapper = audioload.Hdf5iterator{module=dataloader,filepath='myfile',chunksize=10000}
    sys.fexecute("rm myfile")
end

function modeltester:iterwavedataloaderrandom()
    local filepath = wavefilelist
    local dataloader = audioload.WaveDataloader{path=filepath,framesize=100}

    local wrapper = audioload.Hdf5iterator{module=dataloader,filepath='myfile2'}
    -- Prepare the Labels etc.
    local labcount = torch.zeros(dataloader:nClasses())
    local classsizes= torch.Tensor(dataloader:nClasses()):zero()
    local sampletoclass = torch.Tensor(dataloader:size())

    for s,e,inp,lab in wrapper:sampleiterator(1) do
        labcount[lab[1]] = labcount[lab[1]] + 1
        tester:assert(inp:size(1) == lab:size(1))
        sampletoclass[s] = lab[1]
        local addone = torch.Tensor(lab:size(1)):fill(1)
        classsizes:indexAdd(1,lab:long(),addone)
    end

    tester:assert(labcount:sum() == wrapper:size())
    local randomizetol = math.ceil(dataloader:size()/2)

     -- Emulate some 3 iterations over the data
    for i=1,3 do
        local randomized = 0
        local tmpclasssizes = torch.Tensor(dataloader:nClasses()):zero()
        for s,e,inp,lab in dataloader:sampleiterator(1,nil,true) do
            if sampletoclass[s] ~= lab[1] then
                randomized = randomized + 1
            end
            local addone = torch.Tensor(lab:size(1)):fill(1)
            tmpclasssizes:indexAdd(1,lab:long(),addone)
            tester:assert(inp:size(1) == lab:size(1))
        end
        tester:eq(tmpclasssizes,classsizes)
        tester:assert(randomized ~= 0 and randomized > randomizetol,"Randomization not done correctly")
    end
    sys.fexecute("rm myfile2")
end

function modeltester:iterwavedataloader()
    local filepath = wavefilelist
    local dataloader = audioload.WaveDataloader{path=filepath,framesize=100}

    local wrapper = audioload.Hdf5iterator{module=dataloader,filepath='myfile2'}

    local it = wrapper:sampleiterator(128,nil,true)
    local tic = torch.tic()
    for i,k,v,d in it do
        -- print(i,k,v,d)
    end
    print("\nHDF took " .. torch.toc(tic))

    it = dataloader:sampleiterator(128,nil,true)
    tic = torch.tic()
    for i,k,v,d in it do

    end
    print("\nWave took " .. torch.toc(tic))
    sys.fexecute("rm myfile2")
end

function modeltester:benchmarkhtkiter()
    local dataloader = audioload.HtkDataloader(htkfilelist)
    local wrapper = audioload.Hdf5iterator{module=dataloader,filepath='myfile2'}

    local it = wrapper:sampleiterator(128,nil,true)
    local tic = torch.tic()
    for i,k,v,d in it do
        -- print(i,k,v,d)
    end
    print("\nHDF took " .. torch.toc(tic))

    it = dataloader:sampleiterator(128,nil,true)
    tic = torch.tic()
    for i,k,v,d in it do

    end
    print("\nHTK iterator took " .. torch.toc(tic))
    sys.fexecute("rm myfile2")
end

--
function modeltester:iterseqence()
    local filepath = wavefilelist
    local dataloader = audioload.WaveDataloader(filepath,100)

    local seqdataloader = audioload.Sequenceiterator(dataloader,20)

    local wrapper = audioload.Hdf5iterator{module=seqdataloader,filepath='myfile2'}

    local it = wrapper:sampleiterator(128,nil,true)
    local tic = torch.tic()
    for i,k,v,d in it do
    end
    print("\nHDF Took "..torch.toc(tic))

    sys.fexecute("rm myfile2")
end
--

function modeltester:iterutt()
    local filepath = wavefilelist
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

if not wavefilelist or wavefilelist == "" then
    print("Please pass a wave filelist as first argument")
    return
end

tester:add(modeltester)
tester:run()
