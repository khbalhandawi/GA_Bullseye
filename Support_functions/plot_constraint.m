function YX_cstr = plot_constraint(X,Y,YX_cstr,color,h)
    % Function to plot constraint as a hatched region
    cond = (YX_cstr > 0); % replace infeasible region with zeros
    YX_cstr(cond) = 0;
    if all(YX_cstr == 0,'all')
        YX_cstr(1,1) = -0.1; % override controur not rendered warning
    end

    [c1, h1] = contourf(h,X,Y,YX_cstr,[0 0]);
    % if not(all(YX_cstr == 0,'all'))
    if not(all(YX_cstr < 0,'all')) % avoid not rendering when entire space is feasible
        set([h1],'linestyle','-','LineColor',color,'LineWidth',2,'Fill','on','Visible','on','Tag','HatchingRegion');
    end
end

