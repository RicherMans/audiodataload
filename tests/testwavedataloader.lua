require 'torch'
require 'nn'

-- Add to cpath
local info = debug.getinfo(1,'S')
local script_path=info.source:sub(2):gsub("testwavedataloader",'?')
package.path = package.path .. ";".. script_path .. ";../".. script_path..  ";../../" ..script_path
local modeltester = torch.TestSuite()
-- Inits the dataloaders
local audioload = paths.dofile('../init.lua')

local tester = torch.Tester()


function modeltester:init()
    local filepath = "train.lst"
    local dataloader = audioload.WaveDataloader(filepath,100)

    local it = dataloader:sampleiterator(256)
    dataloader:random()
    local timer = torch.Timer()
    for i,k,v,d in it do
        print(timer:time().real)
        timer:reset()
        -- print(i,k,v,d)
    end
end


tester:add(modeltester)
tester:run()
