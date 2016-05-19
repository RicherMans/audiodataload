require 'torch'
require 'nn'

-- Add to cpath
local info = debug.getinfo(1,'S')
local script_path=info.source:sub(2):gsub("testwavedataloader",'?')
package.path = package.path .. ";".. script_path .. ";../".. script_path..  ";../../" ..script_path
local modeltester = torch.TestSuite()
-- Inits the dataloaders
local audioload = paths.dofile('../init.lua')

print(audioload)

local tester = torch.Tester()


function modeltester:init()
    local filepath = ""
    local dataloader = audioload.WaveDataloader()
end


tester:add(modeltester)
tester:run()
