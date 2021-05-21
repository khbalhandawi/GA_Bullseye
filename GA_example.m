%% Preamble
clearvars
close all
clc
format compact

addpath Support_functions
addpath ./Support_functions/hatchfill2_r8
global ax
% rng default % For reproducibility

%% Bullseye Problem (Eosenbrock + concentric circle constraints)
LB = [0,0]; % Lower bound
UB = [2,2]; % Upper bound

x1 = linspace(0,2,100);
x2 = linspace(0,2,100);

[X1,X2] = meshgrid(x1,x2);
X = [reshape(X1,[],1), reshape(X2,[],1)];

% Theoritical minimum
[Y,c] = model(X);
[opt,i] = min(Y);
opt_x = X(i,:);

% plot objective function
Y = reshape(Y,size(X1));
[~,ax] = build_contour(LB,UB);
[~, ~] = contourf(ax, X1, X2, Y); % plot contour

% plot constraint function
c = reshape(c,size(X1));
plot_constraint(X1,X2,c,'r',ax);
commit_hatch(ax) % hatch objects

%% Set up augmented Lagrangian GA optimization
global cumilative_f_evals history budget 
global max_generations_per_it max_stall_generations_per_it min_tol_per_it

cumilative_f_evals = 0; history = [0.5 0.5 0 0]; budget = 10000; 
max_generations_per_it = 300; max_stall_generations_per_it = 10; min_tol_per_it = 1e-3;
nvars = 2; % Number of variables

% Non-default options
% options = optimoptions('ga','MaxStallGenerations',100,...
%     'PopulationSize',30,'NonlinearConstraintAlgorithm','auglag',...
%     'FunctionTolerance',1e-6,'ConstraintTolerance',0,...
%     'InitialPopulationRange',[0;2],'Display','final',...
%     'CrossoverFraction',0.1,'MutationFcn',{@mutationadaptfeasible,1,.5},...
%     'OutputFcn',@callback_function);

% Default options
options = optimoptions('ga','Display','final',...
    'PopulationSize',30,'OutputFcn',@callback_function);

% Use GA optimization
[x,fval] = ga(@combined_function_obj,nvars,[],[],[],[],LB,UB,@combined_function_cstr,options)
scatter(ax,x(1),x(2),'*c','linewidth',10) % Plot final result

%% User defined functions

%=========================================================================%
%                          PLOTTING FUNCTIONS                             %
%=========================================================================%
function [fig,ax] = build_contour(lob,upb)
    % Setup figure
    fig = figure(1);
    ax = gca;

    colormap(ax,'default')
    colorbar(ax)
%     caxis(ax,[min(min(Y)),max(max(Y))])

    axis(ax,[lob(1),upb(1),lob(2),upb(2)]) % fix the axis limits
    hold(ax,'on')

    % plot3(h,S(:,i(1)),S(:,i(2)),Y,'.k', 'MarkerSize',3)
    xlabel(ax,'x')
    ylabel(ax,'y')
    zlabel(ax,'f')
    
end

function [fig,ax1,ax2] = build_fig()
    
    fig = figure(2);
    fig_height = 250*2;
    set(fig, 'Position', [100, 100, 600, 800])
    
    ax1 = subplot(2,1,1,'Parent',fig); % subplot
    % set(ax,'xlim',[1,options.MaxGenerations+1]);
    title('Range of population size, Mean','interp','none')
    xlabel(ax1,'Number of function evaluations','interp','none')
    
    ax2 = subplot(2,1,2,'Parent',fig); % subplot
    % set(ax,'xlim',[1,options.MaxGenerations+1]);
    title(ax2,'Range of population score, Mean','interp','none')
    xlabel(ax2,'Number of function evaluations','interp','none')
    
end

%=========================================================================%
%                           BULLSEYE PROBLEM                              %
%=========================================================================%
function [f,c] = model(x)

    % Objective
    f = log(1 + 100*(x(:,1).^2 - x(:,2)).^2 + (1 - x(:,1)).^2) + 0.5*(rand - 0.5);

    % Constraint
    h = 1; d = 1; r1 = 0.2; r2 = 0.8;
    center = [h,d];
    for i = 1:1:size(x,1)
        dist = norm(x(i,:) - center);

        if dist <= r2 && dist >= r1
            c(i) = 1;
        else
            c(i) = -0.000001;
        end
    end
end

%=========================================================================%
%                           'SMART' FUNCTIONS                             %
%=========================================================================%
function f = combined_function_obj(x)
    global cumilative_f_evals history

    RowIdx = find(ismember(history(:,1:length(x)), x,'rows'));
    
    if isempty(RowIdx) == 0 % If point has been previously evaluated
        
        f = history(RowIdx(end), 3);       
        c = history(RowIdx(end), 4);    
        
    else  % Compute point
        
        [f,c] = model(x);
        
        % Update history and counter
        % fprintf('obj eval point X = [ %f  ,  %f ]\n',x(:,1),x(:,2))
        history = [history; [x, f, c]];
        cumilative_f_evals = 1 + cumilative_f_evals;
    end
    
end

function [c,ceq] = combined_function_cstr(x)
    global cumilative_f_evals history
    
    RowIdx = find(ismember(history(:,1:size(x,2)), x,'rows'));
    
    if isempty(RowIdx) == 0 % If point has been previously evaluated
        
        f = history(RowIdx(end), 3);       
        c = history(RowIdx(end), 4);    
        
    else % Compute point
        
        [f,c] = model(x);
        
        % Update history and counter
        % fprintf('cstr eval point X = [ %f  ,  %f ]\n',x(:,1),x(:,2))
        history = [history; [x, f, c]];
        cumilative_f_evals = 1 + cumilative_f_evals;
    end
    ceq = [];
    
end

%=========================================================================%
%            GA OUTPUT FUNCTION (called after each generation)            %
%=========================================================================%
function [state,options,optchanged] = callback_function(options,state,flag)
    %callback_function Plots the mean and the range of the population.
    %   [state,options,optchanged] = callback_function(OPTIONS,STATE,FLAG) plots the mean and the range
    %   (highest and the lowest distance) of individuals.  
    %   (highest and the lowest score) of individuals. 
    %   Copyright 2021-2020 Khalil Al Handawi
    global ax ax2 ax3 cumilative_f_evals budget % variables for controlling total computational budget
    global max_generations_per_it max_stall_generations_per_it % variables for controlling auglag step
    global h_max stall_G_sub f_best n_sub_generations % variables for keeping track of augmented lagrange step
    optchanged = false;
    
    G = state.Generation;
    population = state.Population;
    score = state.Score;
    
    % Score mean
    smean = nanmean(score); % Ignore infeasible individuals with a NaN score
    Y = smean;
    L = min(score);
    U = max(score);
    
    % Distance mean
    population_center = mean(population,1); % mean along rows
    distances = vecnorm ([population(:,1) - population_center(1), population(:,2) - population_center(2)],2,2);
    Y_m = mean(distances);
    L_m = min(distances);
    U_m = max(distances);

    state.FunEval = cumilative_f_evals;
    
    switch flag

        case 'init'
            fprintf('        G_sub     F_evals     f_best        H_max       G_sub     Tol            (  x_best                )\n')
            fprintf('        ------------------------------------------------------------------------------------------------\n')
            
            % Contour plots
            pop_plot = plot(ax,population(:,1),population(:,2),'.g','markersize',20);
            set(pop_plot,'Tag','pop_plot');
            
            % Progress plots
            [fig,ax2,ax3] = build_fig();
            plotRange = errorbar(ax2,cumilative_f_evals,Y_m, Y_m - L_m, U_m - Y_m);
            score_plot = errorbar(ax3,cumilative_f_evals,Y, Y - L, U - Y);
            
            offset = 0.2*max(abs(U_m - L_m));
            set(ax2,'ylim',[min(L_m) - offset,max(U_m) + offset])
            offset = 0.2*max(abs(U - L));
            set(ax3,'ylim',[min(L) - offset,max(U) + offset])
            
            set(plotRange,'Tag','plot1drange');
            set(score_plot,'Tag','score_plot');
            
            % Initialize global variables
            h_max = 0; stall_G_sub = 0;
            
        case 'iter'
            if isempty(state.Best) == 0
                stall_G = G - state.LastImprovement;    
                fprintf('=============================================================================================================\n')
                fprintf('Iteration     G_sub     F_evals     f_best        H_max         S G      S G_sub   ( x_best                 )\n')
                
                % Extract optimum
                x_best = population(score == state.Best(end),:);
                x_best = x_best(end,:);
                
                % Print current generation results
                x_string = repmat('%-.6f   ',1,size(x_best,2));
                fprintf(['%-3d           %-3d       %-5d      %-+.6f      %+3.6f     %-3d      %-3d       ( ',x_string,' )\n'],...
                    G,n_sub_generations,cumilative_f_evals,f_best,h_max,stall_G,stall_G_sub,x_best)
                fprintf('=============================================================================================================\n')
            end
            
            % for sub iterations 
            fprintf('        G_sub     F_evals     f_best        H_max       G_sub     Tol            ( x_best                 )\n')
            fprintf('        ---------------------------------------------------------------------------------------------------\n')
            
        case 'interrupt'

            %-------------------------------------------------------------%
            % Contour plots
            pop_plot = findobj(get(ax,'Children'),'Tag','pop_plot');
            set(pop_plot,'Xdata',population(:,1),'Ydata',population(:,2));
            
            %-------------------------------------------------------------%
            % Progress plots
            
            plotRange = findobj(get(ax2,'Children'),'Tag','plot1drange');
            newX = [get(plotRange,'Xdata') cumilative_f_evals];
            newY = [get(plotRange,'Ydata') Y_m];
            newL = [get(plotRange,'Ldata') (Y_m - L_m)];
            newU = [get(plotRange,'Udata') (U_m - Y_m)];   
            set(get(ax2,'xlabel'),'String','Function evaluations')
            set(get(ax2,'ylabel'),'String','Average population distance')
            set(plotRange,'Xdata',newX,'Ydata',newY,'Ldata',newL,'Udata',newU);
            lower_abs = newY - newL; upper_abs = newY + newU;
            offset = 0.2*max(abs(upper_abs - lower_abs));
            set(ax2,'ylim',[min(lower_abs) - offset,max(upper_abs) + offset])
            
            if isempty(state.Best) == 0 && ~isnan(Y)
                score_plot = findobj(get(ax3,'Children'),'Tag','score_plot');
                newX = [get(score_plot,'Xdata') cumilative_f_evals];
                newY = [get(score_plot,'Ydata') Y];
                newL = [get(score_plot,'Ldata') (Y - L)];
                newU = [get(score_plot,'Udata') (U - Y)];     
                set(get(ax3,'xlabel'),'String','Function evaluations')
                set(get(ax3,'ylabel'),'String','Score')
                set(score_plot,'Xdata',newX,'Ydata',newY,'Ldata',newL,'Udata',newU);
                lower_abs = newY - newL; upper_abs = newY + newU;
                offset = 0.2*max(abs(upper_abs - lower_abs));
                set(ax3,'ylim',[min(lower_abs) - offset,max(upper_abs) + offset])
            end
            
            %-------------------------------------------------------------%
            % Reset sub generation variables after an iteration is completed
            if isempty(state.Best) == 0
                f_best = state.Best(end);
                stall_G_sub = G - state.LastImprovement;
                n_sub_generations = G;
                
                % get current tolerance
                tol = abs(diff(state.Best));
                if isempty(tol) == 0
                    current_tol = tol(end);
                else
                    current_tol = 1e9;
                end

                %---------------------------------------------------------%
                % Extract optimum
                x_best = population(score == state.Best(end),:);
                x_best = x_best(end,:);
                
                % Print current generation results if new optimum is found
                if stall_G_sub == 0
                    x_string = repmat('%-.6f   ',1,size(x_best,2));
                    fprintf(['        %-3d       %-5d       %-+.6f     %+3.6f   %-3d       %-.6e   ( ',x_string,' )\n'],...
                        n_sub_generations,cumilative_f_evals,f_best,h_max,stall_G_sub,current_tol,x_best)
                    
                    shg(); % show current figure
                end

            else
                state.LastImprovement = G+1;
            end
            


            %-------------------------------------------------------------%
            % Stopping criteria 
            if cumilative_f_evals >= budget
                state.StopFlag = ['computational budget of ',num2str(budget), ' reached'];
            end
            
            % if G > 0
            %     if G > max_generations_per_it || stall_G_sub > max_stall_generations_per_it
            %         % state.StopFlag = ['computational budget of ',num2str(budget), ' reached'];
            %         state.Generation = G + options.MaxStallGenerations;
            %         state.Best(state.Generation) = state.Best(end);
            %     end
            % end
            
    end
    
    if ~strcmp(flag,'interrupt')
        h_max = state.NonlinIneq;
    end
    
end