function commit_hatch(h)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    hg1onlyopt = {};
    hp = findobj(h,'Tag','HatchingRegion');
    try
        hatchfill2(hp,'single','LineWidth',2,hg1onlyopt{:},'Fill','on','HatchVisible','on');
    catch
        fprintf('No area to hatch in constraint')
    end
end

