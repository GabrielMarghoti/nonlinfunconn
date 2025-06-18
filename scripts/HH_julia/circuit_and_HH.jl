#
# Hodgkin-Huxley Neuron model Nonequilibrium Green Functions
#
# Gabriel Marghoti / 2024 - UFPR - Curitiba - Brazil
#
using Plots, Colors
using DifferentialEquations
using LaTeXStrings
using ProgressBars
using Statistics
using Measures


gr() 
default(fontfamily="Computer Modern", linewidth=2, label=nothing, grid=false, framestyle=:box)


function theta(x)
    if x<0
        return 0.0
    else
        return 1.0
    end
end

function alpha_S(V, a_r, beta, V_th)
    return a_r/(1 + exp(-beta*(V-V_th)))
end
function d_alpha_S(V, a_r,  beta, V_th)
    return (a_r*beta*exp(-beta*(V-V_th))/(1 + exp(-beta*(V-V_th)))^2)
end


function conv(X, Y, dt)
    # Trapezoidal rule for discrete convolution
    n = length(X)
    s = zero(eltype(X))
    for i in 1:n
        if i == 1 || i == n
            s += 0.5 * (X[i] * Y[i])
        else
            s += (X[i] * Y[i])
        end
    end
    return dt * s
end
#=
alpha_m(V) = 0.1 * (V+25.0) / (exp((V+25.0)/10) - 1.0)
beta_m(V) = 4.0 * exp((V) / 18.0)

alpha_h(V) = 0.07 * exp((V) / 20.0)
beta_h(V) = 1.0 / (exp((V+30.0)/10) + 1.0)

alpha_n(V) = 0.01 * (V+10.0) / (exp((V+10.0)/10) - 1.0)
beta_n(V) = 0.125 * exp(V / 80.0)
=#

alpha_m(V) = 0.1 * (V+40.0) / (1.0 - exp(-0.1 * (V+40.0)))
beta_m(V) = 4.0 * exp(-(V+65.0) / 18.0)
alpha_h(V) = 0.07 * exp(-(V+65.0) / 20.0)
beta_h(V) = 1.0 / (1.0 + exp(-0.1 * (V+35.0)))
alpha_n(V) = 0.01 * (V+55.0) / (1.0 - exp(-0.1 * (V+55.0)))
beta_n(V) = 0.125 * exp(-(V+65.0) / 80.0)

function create_rgba_code_line(n_lines)
    # Generate n_lines colors from the jet colormap
    jet_colors = cgrad(:jet, n_lines) # or :phase

    # Create an array to store RGBA values (10x4 matrix)
    rgba_matrix = zeros(Float64, n_lines, 4)

    # Fill the matrix with RGBA values
    for i in 1:n_lines
        color = jet_colors[Int(i*div(length(jet_colors), n_lines))] 
        rgba_matrix[i, :] = [red(color), green(color), blue(color), alpha(color)]
    end

    return rgba_matrix
end
# Hodgkin-Huxley model equations
function hh_model!(du, u, p, t)
    
    # Load parameters in the integrator step function
    C_m, g_Na_bar, g_K_bar, g_L_bar, E_Na, E_K, E_L, N, A, B, Es, a_r, a_d, beta, V_th, I_ext,  stim_amplitude, stim_starts, stim_ends = p

    for i=1:N # Iteration over neurons variables
        V, m, h, n = u[i, 1:4]

        I_ext = 0.0
        for stim_idx in 1:size(stim_starts, 2)
            t_start = stim_starts[i, stim_idx]
            t_end = stim_ends[i, stim_idx]
            if !isnan(t_start) && !isnan(t_end)
                if t >= t_start && t <= t_end
                    I_ext += stim_amplitude
                end
            end
        end

        for j=1:N # Coupling terms from pre-synaptic neurons
            I_ext += -A[i, j]*(u[i,1] - u[j,1]) - B[i,j]*u[i,4+j]*(u[i, 1] - Es[i,j])
            du[i, 4+j] = alpha_S(u[j,1], a_r[i,j], beta[i,j], V_th[i,j])*(1-u[i, 4+j])-a_d[i,j]*u[i, 4+j]
        end
        
        du[i, 1] = (I_ext - g_Na_bar*m^3*h*(V - E_Na) - g_K_bar*n^4*(V - E_K) - g_L_bar*(V - E_L)) / C_m
        du[i, 2] = alpha_m(V) * (1.0 - m) - beta_m(V) * m
        du[i, 3] = alpha_h(V) * (1.0 - h) - beta_h(V) * h
        du[i, 4] = alpha_n(V) * (1.0 - n) - beta_n(V) * n
    end
end


function main()
    
    # Time span
    ti =   0.0
    tf =  150.0
    
    resolution = 1500

    tspan = (ti, tf)

    ts = range(ti, tf, length=resolution)

    dt = ts[2] - ts[1]

    # Neuron Parameters
    # Define constants
    C_m      =   1.0         # membrane capacitance, in uF/cm^2
    g_Na_bar =  120.0         # maximum conductances, in mS/cm^2
    g_K_bar  =  36.0
    g_L_bar  =   0.5
    E_Na     =  50.0         # Nernst reversal potentials, in mV
    E_K      = -77.0
    E_L      = -55.0 

    # Synapses parameters
    N    = 3 # Number of neurons
    
    # Coupling matrices, adjacency matrices
    A    = [  0.0  0.0   0.0;  # S/F   # Gap junction coupling  
              0.0  0.0   0.0;
              0.0  0.0   0.0]   

    B    = [  0.0  0.0   0.0;  # S/F    # Chemical synapse coupling
              0.0  0.0   0.0;
              0.2  0.2   0.0] 

    Es   = fill(0.0, (N,N))        # mV  # Synapse Nerst potential, determines excitation or inhibition 
    #Es[3, 2] = -70                # Change this to control Inhibition(-70 value) depends on the synaptic receptor ion potential
    a_r  = fill(5.0, (N,N))        # rates for synaptic channel conductance activity  
    a_d  = fill(5.0, (N,N))
    #a_r[4, 1] = 1.0
    beta    = fill(0.125, (N,N))      # (mV)⁻¹
    V_th = fill(-50.0  , (N,N))    #  mV    # update when having the equilibrium values
    #V_th[3, 2] = -10.0             #  mV    # alpha_S(V=V_th) = 1/2

    # External current
    I_ext  =  0.0  # in μA/cm^2     
    stim_amplitude  =  10.0  # in μA/cm^2    

    It = zeros(resolution, N)
    # Define stimulation windows for each neuron as a dictionary
    stim_starts = [
        2.0   60.0  114.0;
        2.0   64.0  110.0;
        NaN   NaN   NaN
    ]
    stim_ends = [
        5.0   63.0  117.0;
        5.0   67.0  113.0;
        NaN   NaN   NaN
    ]
    # Fill It using stim_starts and stim_ends matrices
    for neuron in 1:N
        for stim_idx in 1:size(stim_starts, 2)
            t_start = stim_starts[neuron, stim_idx]
            t_end = stim_ends[neuron, stim_idx]
            if !isnan(t_start) && !isnan(t_end)
                for t = 1:resolution
                    if ts[t] >= t_start && ts[t] <= t_end
                        It[t, neuron] = stim_amplitude
                    end
                end
            end
        end
    end

    ts_plot = [50; 80; 100; 120; 150; 180; 200; 220; 250; 300; 350; 400]

    figures_path = "figures/HH_model_PHDqualification/NEGF_HH_AND_circuit_$(N)neurons_tf$(tf)/$(g_Na_bar)_$(g_K_bar)_$(g_L_bar)_$(E_Na)_$(E_K)_$(E_L)_syn_$(minimum(Es))_stim_$(I_ext)_$(stim_amplitude))/"
    
    for i=1:N
        mkpath(figures_path*"/neuron$(i)/")
    end

    u0 = rand(N, N+4)

    V0 = zeros(N)
    m0 = zeros(N)
    h0 = zeros(N)
    n0 = zeros(N)

    gS0 = zeros(N, N)

    for itr=1:5 #iteraction to remove transient and set parameter V_th equals to the equilibrium values V0
    # Solve the differential equations TRANSIENT
        prob = ODEProblem(hh_model!, u0, tspan, [C_m, g_Na_bar, g_K_bar, g_L_bar, E_Na, E_K, E_L, N, A, B, Es, a_r, a_d, beta, V_th, I_ext, 0.0, stim_starts, stim_ends])
        sol = solve(prob, Tsit5(),  dt=0.01, saveat=ts, reltol=1e-9, abstol=1e-9, maxiters = 1e7)

    #    png(plot(sol, layout=(4,1), lc=:black, xlabel=["" "" "" L"Time \ (ms)"], ylabel=["V(t) (mV)" "m(t)" "h(t)" "n(t)"], label="", frame_style=:box, size=(500,500), dpi=200), figures_path*"neuron$(i)/HH_variables_simulation_transient")

        u0 = sol[end]
        
        for i=1:N
            V0[i], m0[i], h0[i], n0[i] = u0[i,1:4]
        end
        
        gS0 = u0[:, 5:end]

        V_th = [fill(V0[1], N) fill(V0[2], N) fill(V0[3], N)] 
        
    end


    # Solve the differential equations STIMULATED
    prob = ODEProblem(hh_model!, u0, tspan, [C_m, g_Na_bar, g_K_bar, g_L_bar, E_Na, E_K, E_L, N, A, B, Es, a_r, a_d, beta, V_th, I_ext, stim_amplitude, stim_starts, stim_ends])
    sol = solve(prob, Euler(), dt=0.01, saveat=ts, reltol=1e-9, abstol=1e-9, maxiters=1e7)

    V = zeros(resolution, N)
    m = zeros(resolution, N)
    h = zeros(resolution, N)
    n = zeros(resolution, N)

    gS = zeros(resolution, N, N)

    for t=1:resolution
        V[t, :]     = sol[t][:, 1]
        m[t, :]     = sol[t][:, 2]
        h[t, :]     = sol[t][:, 3]
        n[t, :]     = sol[t][:, 4]
        gS[t, :, :] = sol[t][:, 5:end]
    end
    
    DeltaV = zeros(resolution, N)
    DeltagS = zeros((resolution, N, N))

    for t=1:resolution
        DeltaV[t, :] = V[t, :] .- V0
        DeltagS[t, :, :] = gS[t, :, :] .- gS0
    end

    n_inf = alpha_n.(V) ./ (alpha_n.(V) .+ beta_n.(V))
    tau_n = 1. ./ (alpha_n.(V) .+ beta_n.(V))
    n_inf0 = alpha_n.(V[1, :]) ./ (alpha_n.(V[1, :]) .+ beta_n.(V[1, :]))

    m_inf = alpha_m.(V) ./ (alpha_m.(V) .+ beta_m.(V))
    tau_m = 1. ./ (alpha_m.(V) .+ beta_m.(V))
    m_inf0 = alpha_m.(V[1, :]) ./ (alpha_m.(V[1, :]) .+ beta_m.(V[1, :]))

    h_inf = alpha_h.(V) ./ (alpha_h.(V) .+ beta_h.(V))
    tau_h = 1. ./ (alpha_h.(V) .+ beta_h.(V))
    h_inf0 = alpha_h.(V[1, :]) ./ (alpha_h.(V[1, :]) .+ beta_h.(V[1, :]))


    tau_i = zeros(N)
    
    for i=1:N
        tau_i[i] = (C_m)/(g_L_bar+(g_Na_bar*m0[i]^3)*h0[i]+(g_K_bar*n0[i]^4)+ sum(A[i,:]) + sum(B[i,:].*gS0[i,:]))
    end
    println("tau_i = ", tau_i)

    plot_Vs =  plot(ts, V, layout=(N,1), 
    lc=:black, xlabel=[ "" "" "Time (ms)"], ylabel=[ L"\Delta V(t)" L"\Delta V(t)" L"\Delta V(t)"], label=["External input" "Neuron 1" "Neuron 2" "Neuron 3"], 
    frame_style=:box, size=(500, 200*N+20), dpi=200, grid=false)
    plot!(ts, It,layout=(N,1), lc=:red, label="I(t)", subplot=1, xlabel="Time (ms)")

    png(plot_Vs, figures_path*"Vs_variables_simulation_neurons")
    
    plot_Delta_Vs =  plot(ts, DeltaV,  layout=(N,1), 
    lc=:black, xlabel=[ "" "" "Time (ms)"], ylabel=[L"I(t)" L"\Delta V(t)" L"\Delta V(t)" L"\Delta V(t)"], label=["External input" "Neuron 1" "Neuron 2" "Neuron 3"], 
    frame_style=:box, size=(500, 200*N+20), dpi=200, grid=false)
    plot!(ts, It, layout=(N,1),lc=:red, label="I(t)", subplot=1, xlabel="Time (ms)")
    
    png(plot_Delta_Vs, figures_path*"DeltaVs_variables_simulation_neurons")

#=
    for i=1:N
        var_plots = plot(sol.t, [V[:, i] m[:, i] h[:, i] n[:, i]], layout=(4,1), 
        lc=:black, xlabel=["" "" "" L"Time \ (ms)"], ylabel=["V(t) (mV)" "m(t)" "h(t)" "n(t)"], label="", 
        frame_style=:box, size=(900,500), dpi=200, grid=false)
        hline!([V0[i] m0[i] h0[i] n0[i]], label="", lc=:black, ls=:dot)
        
        png(var_plots, figures_path*"neuron$(i)/HH_variables_simulation_neurons")
            
        var_syn_plots = plot(sol.t, [gS[:, i,1] gS[:, i,2]], layout=(4,1), 
            lc=:black, xlabel=["" L"Time \ (ms)"], ylabel=["gS$(i)1" "gS$(i)2"], label="", 
            frame_style=:box, size=(900,500), dpi=200, grid=false)
            hline!([gS0[i,1] gS0[i,2]], label="", lc=:black, ls=:dot)
        
        png(var_syn_plots,figures_path*"neuron$(i)/Synaptic_var_gS_post_syn")


        cond_plote = plot(ts, [V[:, i] (g_Na_bar*m[:, i].^(3).*h[:, i]) (g_K_bar*n[:, i].^4)], layout=(3,1), 
        lc=:black, xlabel=["" "" L"Time \ (ms)"], ylabel=["V(t)(mV)" "g_{Na} (t)" "g_{K} (t)"], label="", 
        frame_style=:box, size=(900,500), dpi=200, left_margin=5mm, grid=false)
        hline!([V0[i] (g_Na_bar*m0[i].^(3).*h0[i]) (g_K_bar*n0[i].^4)], label=["Equilibrium" "" "" ""], lc=:red, ls=:dot)
        
        png(cond_plote, figures_path*"neuron$(i)/HH_conductances_simulation")

        png(plot(ts, [(tau_n[:, i]) (tau_m[:, i]) (tau_h[:, i])], layout=(3,1), 
        lc=:black, xlabel=["" "" "Time(ms)"], ylabel=["n" "m" "h"], label="", 
        frame_style=:box, size=(500,500), dpi=200, grid=false),
        figures_path*"neuron$(i)/HH_taus")

        png(plot(ts, [(n_inf[:, i]) (m_inf[:, i]) (h_inf[:, i])], layout=(3,1), 
        lc=:black, xlabel=["" "" "Time(ms)"], ylabel=["n" "m" "h"], label="", 
        frame_style=:box, size=(500,500), dpi=200, grid=false),
        figures_path*"neuron$(i)/HH_saturation variables")
    end
=#
    #### Equilibrium effective green functions
    sigma0  = zeros(resolution, resolution, N, N)
    gg0 = zeros(resolution, resolution, N, N)
    gs0 = zeros(resolution, resolution, N, N)
    g0  = zeros(resolution, resolution, N, N)

    for j=1:N
        for i=1:N
            for t =1:resolution
                for t′=1:resolution
                    sigma0[t, t′, i, j]  = theta(ts[t]-ts[t′]) *(1-gS0[i, j])*d_alpha_S(V0[j], a_r[i,j], beta[i,j], V_th[i,j])*exp(-(ts[t]-ts[t′])*(a_d[i,j]+alpha_S(V0[j], a_r[i,j], beta[i,j], V_th[i,j])))
                    
                    gg0[t, t′, i, j] = theta(ts[t]-ts[t′]) *A[i, j]*exp(-(ts[t]-ts[t′])/(tau_i[i]))/C_m
                    gs0[t, t′, i, j] = theta(ts[t]-ts[t′]) *B[i, j]*(Es[i,j]-V0[i])*exp(-(ts[t]-ts[t′])/(tau_i[i]))/C_m
                end
            end
            
            #=png(
                heatmap(ts, ts, sigma0[:, :, i, j],
                    xflip=false,
                    ylabel="t (current time)",
                    xlabel = "t′ (past time)",
                    size=(500,400),
                    fillcolor=:jet1,
                    dpi=200, 
                    frame_style=:box,
                    grid=false),
                figures_path*"neuron$(i)/sigmaZERO_ij$(i)_$(j)"
            )=#
#=
            png(heatmap(ts, ts, gs0[:, :, i, j],
            xflip=false,
            ylabel="t (current time)",
            xlabel = "t′ (past time)",
            size=(500,400),
            fillcolor=:jet1,
            dpi=200, 
            frame_style=:box,
            grid=false),
                figures_path*"neuron$(i)/gsZERO_ij$(i)_$(j)"
            )

            png(heatmap(ts, ts, gg0[:, :, i, j],
            xflip=false,
            ylabel="t (current time)",
            xlabel = "t′ (past time)",
            size=(500,400),
            fillcolor=:jet1,
            dpi=200, 
            frame_style=:box,
            grid=false),
                figures_path*"neuron$(i)/ggZERO_ij$(i)_$(j)"
            )=#
        end
    end

    for t=1:resolution
        for t′=1:(t)
            for i=1:N
                for j=1:N
                    g0[t, t′, i, j] = gg0[t, t′, i, j] + conv(gs0[t, t′:t, i, j], sigma0[t′:t, t′, i, j], dt)
                end
            end
        end
    end


#   NONequilibrium green functions
    sigma_n_V  = zeros(resolution, resolution, N)
    kappa_gK_V = zeros(resolution, resolution, N)
    
    sigma_m_V  = zeros(resolution, resolution, N)
    sigma_h_V  = zeros(resolution, resolution, N)
    kappa_gNa_V = zeros(resolution, resolution, N)
    
    conv_sigma_n_V = zeros(resolution, N) # n approx
    conv_sigma_m_V = zeros(resolution, N) # m approx
    conv_sigma_h_V = zeros(resolution, N) # h approx


    conv_kappa_K_V  = zeros(resolution, N) 
    conv_kappa_Na_V = zeros(resolution, N) 
    
    Chi_K  = zeros(resolution, resolution, N)
    Chi_Na = zeros(resolution, resolution, N)

    Chi_i  = zeros(resolution, resolution, N)

    conv_chi_V_i = zeros(resolution, N)
    conv_gamma_V_i = zeros(resolution, N)

    conv_exptau_I = zeros(resolution, N)

    signal_components = zeros(resolution, N, N)

    ext_comp = zeros(resolution, N)

    for i=1:N
        for t=1:resolution  # external stim only to the first neurons
            conv_exptau_I[t, i] =  conv(exp.(-(ts[t].-ts[1:t])/(tau_i[i]))/C_m, It[1:t, i], dt)
        end
    end
    for itr=ProgressBar(1:5)  #### iterative method to approximate sigma
        for i=1:N
            for t=1:resolution
                for t′=1:(t) 
                    if (DeltaV[t′, i] != 0.0)
                        # Potassium Channel
                        sigma_n_V[t, t′, i]  = (n_inf[t′, i]- (n_inf0[i] + conv_sigma_n_V[t′, i])) / (tau_n[t′, i]*(DeltaV[t′, i]))
                        kappa_gK_V[t, t′, i] = 4*g_K_bar*((conv_sigma_n_V[t′, i]+n0[i])^3)*(n_inf[t′, i]- n0[i] - (conv_sigma_n_V[t′, i]))/(tau_n[t′, i]*(DeltaV[t′, i]))

                        # Sodium Channel
                        sigma_m_V[t, t′, i]  = (m_inf[t′, i].- (m_inf0[i] .+ conv_sigma_m_V[t′, i])) / (tau_m[t′, i]*(DeltaV[t′, i]))
                        sigma_h_V[t, t′, i]  = (h_inf[t′, i].- (h_inf0[i] .+ conv_sigma_h_V[t′, i])) / (tau_h[t′, i]*(DeltaV[t′, i]))
                        kappa_gNa_V[t, t′, i] = (g_Na_bar*(conv_sigma_m_V[t′, i]+m0[i])^2)*(3*(conv_sigma_h_V[t′, i]+h0[i])*(((m_inf[t′, i]) - (conv_sigma_m_V[t′, i]+m0[i]))/(tau_m[t′, i]*(DeltaV[t′, i]))) + (conv_sigma_m_V[t′, i]+m0[i])*(((h_inf[t′, i]) - (conv_sigma_h_V[t′, i]+h0[i]))/(tau_h[t′, i]*(DeltaV[t′, i]))))
                    end
                    Chi_K[t, t′, i]  = -exp(-(ts[t]-ts[t′])/(tau_i[i]))*(V[t′, i] - E_K )/C_m
                    Chi_Na[t, t′, i] = -exp(-(ts[t]-ts[t′])/(tau_i[i]))*(V[t′, i] - E_Na)/C_m
                end
                conv_sigma_n_V[t, i] = conv(sigma_n_V[t, 1:t, i], DeltaV[1:t, i], dt)
                conv_sigma_m_V[t, i] = conv(sigma_m_V[t, 1:t, i], DeltaV[1:t, i], dt)
                conv_sigma_h_V[t, i] = conv(sigma_h_V[t, 1:t, i], DeltaV[1:t, i], dt)
            end
            
            for t=1:resolution
                for t′=1:(t)
                    Chi_i[t, t′, i] = conv(Chi_K[t, t′:t, i], kappa_gK_V[t′:t, t′, i], dt) + conv(Chi_Na[t, t′:t, i], kappa_gNa_V[t′:t, t′, i], dt)  
                end
                conv_chi_V_i[t, i]   =  conv(Chi_i[t, 1:t, i], DeltaV[1:t, i], dt)

                conv_kappa_K_V[t, i]  = conv(kappa_gK_V[t,  1:t, i], DeltaV[1:t, i], dt)
                conv_kappa_Na_V[t, i] = conv(kappa_gNa_V[t,  1:t, i], DeltaV[1:t, i], dt)
            end
           
            # Plot DeltaV and its convolution with Chi_i for each neuron
            plot_Vs_with_conv_chi_V_i = plot(layout=(N,1), size=(500, 200*N+20), dpi=200, frame_style=:box, grid=false)
            for ni in 1:N
                plot!(plot_Vs_with_conv_chi_V_i, ts, DeltaV[:, ni], lc=:black, label="ΔV$(ni)(t)", subplot=ni, xlabel=(ni==N ? "Time (ms)" : ""), ylabel="Neuron $(ni)")
                plot!(plot_Vs_with_conv_chi_V_i, ts, conv_chi_V_i[:, ni], lc=:red, ls=:dot, label=L"(\chi_i \ast \Delta V_i)(t)", subplot=ni)
            end
            png(plot_Vs_with_conv_chi_V_i, figures_path*"DeltaVs_with_conv_ChiVs_variables_simulation_neurons")

            plot_Vs_minus_conv_chi_V_i =  plot(ts, DeltaV .- conv_chi_V_i, layout=(N,1), 
            lc=:black, xlabel=["" "" L"Time \ (ms)"], ylabel=L"\Delta V_i(t) - (\chi_i \ast \Delta V_i)(t)", label=["Neuron 1" "Neuron 2" "Neuron 3"], 
            frame_style=:box, size=(500, 200*N+20), dpi=200, grid=false)

            png(plot_Vs_minus_conv_chi_V_i, figures_path*"DeltaVs_minus_conv_ChiVs_variables_simulation_neurons")


            plot_conv_chi_V_i =  plot(ts, conv_chi_V_i, layout=(N,1), 
            lc=:black, xlabel=["" "" L"Time \ (ms)"], ylabel=L"(\chi_i \ast \Delta V_i)(t)", label=["Neuron 1" "Neuron 2" "Neuron 3"], 
            frame_style=:box, size=(500, 200*N+20), dpi=200, grid=false)
            
            png(plot_conv_chi_V_i, figures_path*"conv_ChiVs_variables_simulation_neurons")
            #
                png(
                    heatmap(ts, ts, Chi_i[:, :, i],
                            xflip=false,
                            ylabel="t (ms)",
                            xlabel = "t′ (ms)",
                            size=(500,400),
                            title= "Chiᵢ(t,t′)",
                            fillcolor=:jet1,
                            dpi=200, 
                            frame_style=:box,
                            clims = (-1.0,1.0),
                            right_margin=2mm,
                            grid=false),
                            figures_path*"neuron$(i)/Chi_i"
                    )

                plote = plot(ts[ts_plot[1]:(end)], Chi_i[end, ts_plot[1]:(end), i],
                    ylabel= L"χᵢ(t,t′)",
                    label="t=$(round(ts[end],digits=1))",
                    xlabel="t′",
                    lw=2.0,
                    lc=[:black],
                    ls=[:solid],
                    size=(900,400),
                    dpi=200,
                    legend=:outerright,
                    frame_style=:box,
                    ylims=(-10, 10),
                    grid=false)
                    
                    vline!([4; 5],
                    lw=1,
                    la=1.0,
                    label="",
                    lc=[:black],
                    ls=[:dot],
                    annotation=(1.6, 3, text("Stimulus", 12, :black))
                    )
                colors_codes = create_rgba_code_line(length(ts_plot))
                
                for t_idx in 1:length(ts_plot)
                    plot!(ts[1:(ts_plot[t_idx])], Chi_i[ts_plot[t_idx], 1:ts_plot[t_idx], i],
                        ylabel= L"χᵢ(t,t′)",
                        label="t=$(round(ts[ts_plot[t_idx]],digits=1))",
                        xlabel = "t′ (ms)",
                        lc=RGBA(colors_codes[t_idx, 1], colors_codes[t_idx, 2], colors_codes[t_idx, 3], colors_codes[t_idx, 4]),
                        ls=:solid,
                        lw=1.4,
                        size=(900,400),
                        dpi=200,
                        legend=:outerright,
                        frame_style=:box,
                        #xlims=(0,ts[end]),
                        #left_margin=2mm,
                        grid=false)
                end
                png(plote,figures_path*"neuron$(i)/level_CHI_i")


                plote = plot(ts[ts_plot[1]:(end)], Chi_i[end, ts_plot[1]:(end), i],
                    ylabel= L"χᵢ(t,t′)",
                    label="t=$(round(ts[end],digits=1))",
                    xlabel="t′",
                    lw=2.0,
                    lc=[:black],
                    ls=[:solid],
                    size=(900,400),
                    dpi=200,
                    legend=:outerright,
                    frame_style=:box,
                    ylims=(-1, 1),
                    grid=false)
                    
                    vline!([4; 5],
                    lw=1,
                    la=1.0,
                    label="",
                    lc=[:black],
                    ls=[:dot],
                    annotation=(1.6, 0.5, text("Stimulus", 12, :black))
                    )
                colors_codes = create_rgba_code_line(length(ts_plot))
                
                for t_idx in 1:length(ts_plot)
                    plot!(ts[1:(ts_plot[t_idx])], Chi_i[ts_plot[t_idx], 1:ts_plot[t_idx], i],
                        ylabel= L"χᵢ(t,t′)",
                        label="t=$(round(ts[ts_plot[t_idx]],digits=1))",
                        xlabel = "t′ (ms)",
                        lc=RGBA(colors_codes[t_idx, 1], colors_codes[t_idx, 2], colors_codes[t_idx, 3], colors_codes[t_idx, 4]),
                        ls=:solid,
                        lw=1.4,
                        size=(900,400),
                        dpi=200,
                        legend=:outerright,
                        frame_style=:box,
                        #xlims=(0,ts[end]),
                        #left_margin=2mm,
                        grid=false)
                end
                
                png(plote,figures_path*"neuron$(i)/level_CHI_i_zoomIN")

                png(
                    heatmap(ts, ts, sigma_n_V[:, :, i],
                            xflip=false,
                            ylabel="t (current time)",
                            xlabel = "t′ (past time)",
                            size=(500,400),
                            title="sigma_{n,V}(t,t′)",
                            fillcolor=:jet1,
                            dpi=200, 
                            frame_style=:box,
                            clims = (-0.1,0.1),
                            grid=false),
                            figures_path*"neuron$(i)/sigma_n_V"
                    )
                    png(
                        heatmap(ts, ts, kappa_gK_V[:, :, i],
                                xflip=false,
                                ylabel="t (current time)",
                                xlabel = "t′ (past time)",
                                size=(500,400),
                                title="kappa_{K,V}(t,t′)",
                                fillcolor=:jet1,
                                dpi=200, 
                                frame_style=:box,
                                clims = (-0.1,0.1),
                                grid=false),
                                figures_path*"neuron$(i)/Gamma_n_V"
                        )

                png(
                    heatmap(ts, ts, sigma_m_V[:, :, i],
                            xflip=false,
                            ylabel="t (current time)",
                            xlabel = "t′ (past time)",
                            size=(500,400),
                            title="sigma_{m,V}(t,t′)",
                            fillcolor=:jet1,
                            dpi=200, 
                            frame_style=:box,
                            clims = (-0.1,0.1),
                            grid=false),
                            figures_path*"neuron$(i)/sigma_m_V"
                    )
        
                png(
                    heatmap(ts, ts, sigma_h_V[:, :, i],
                            xflip=false,
                            ylabel="t (current time)",
                            xlabel = "t′ (past time)",
                            size=(500,400),
                            title="sigma_{h,V}(t,t′)",
                            fillcolor=:jet1,
                            dpi=200, 
                            frame_style=:box,
                            clims = (-0.1,0.1),
                            grid=false),
                            figures_path*"neuron$(i)/sigma_h_V"
                    )

                png(
                    plot(ts[end] .- ts, sigma_n_V[end, 1:end, i],
                            xflip=false,
                            ylabel= ["sigma_{n,V}(t,t′)"],
                            label="",
                            xlabel = "t - t′",
                            title="t=$(ts[end])",
                            size=(1000,800),
                            lc=[:black :red],
                            ls=[:solid :dot],
                            dpi=200, 
                            frame_style=:box,
                            #ylims = (-0.1,0.1),
                            grid=false),
                            figures_path*"neuron$(i)/level_curve_sigma_n_V"
                    )
                    png(
                        plot(ts[end] .- ts, [kappa_gK_V[end, 1:end, i] ],
                                xflip=false,
                                ylabel= ["kappa_{K,V}(t,t′)"],
                                label="",
                                xlabel = "t - t′",
                                title="t=$(ts[end])",
                                size=(1000,800),
                                lc=[:black :red],
                                ls=[:solid :dot],
                                dpi=200, 
                                frame_style=:box,
                                #ylims = (-0.1,0.1),
                                grid=false),
                                figures_path*"neuron$(i)/level_curve_Gamma_K_V"
                        )
                        png(
                            plot(ts[end] .- ts, [kappa_gNa_V[end, 1:end, i] ],
                                    xflip=false,
                                    ylabel= ["kappa_{Na,V}(t,t′)"],
                                    label="",
                                    xlabel = "t - t′",
                                    title="t=$(ts[end])",
                                    size=(1000,800),
                                    lc=[:black :red],
                                    ls=[:solid :dot],
                                    dpi=200, 
                                    frame_style=:box,
                                    #ylims = (-0.1,0.1),
                                    grid=false),
                                    figures_path*"neuron$(i)/level_curve_Gamma_Na_V"
                            )
                    png(
                        plot(ts, [sigma_m_V[end, 1:end, i] ],
                                xflip=false,
                                ylabel= ["sigma_{m,V}(t,t′)"],
                                label="",
                                xlabel = "t - t′",
                                title="t=$(ts[end])",
                                size=(1000,800),
                                lc=[:black :red],
                                ls=[:solid :dot],
                                dpi=200, 
                                frame_style=:box,
                                ylims = (-0.1,0.1),
                                grid=false),
                                figures_path*"neuron$(i)/level_curve_sigma_m_V"
                        )
                        png(
                            plot(ts[end] .- ts, [sigma_m_V[end, 1:end, i] ],
                                    xflip=false,
                                    ylabel= ["sigma_{h,V}(t,t′)"],
                                    label="",
                                    xlabel = "t - t′",
                                    title="t=$(ts[end])",
                                    size=(1000,800),
                                    lc=[:black :red],
                                    ls=[:solid :dot],
                                    dpi=200, 
                                    frame_style=:box,
                                    ylims = (-0.1,0.1),
                                    grid=false),
                                    figures_path*"neuron$(i)/level_curve_sigma_h_V"
                            )
                plot_conv_K_V = plot(ts, [g_K_bar*n[:, i].^4 conv_kappa_K_V[:, i].+g_K_bar*n0[i]^4],     
                        label=["Original" "Estimated"],
                        size=(500,400),
                        lc = [:black :red],
                        ls = [:solid :dot],
                        xlabel=L"Time \ (ms)",
                        ylabel = "g_K",
                        dpi=200, 
                        frame_style=:box,
                        grid=false)
                    png(plot_conv_K_V,       
                        figures_path*"neuron$(i)/plot_gK_simulated_estimated_itr$(itr)"
                    )

                plot_conv_Na_V = plot(ts, [g_Na_bar*m[:, i].^3 .*h[:, i] conv_kappa_Na_V[:, i].+g_Na_bar*m0[i]^3*h0[i]],     
                        label=["Original" "Estimated"],
                        size=(500,400),
                        lc = [:black :red],
                        ls = [:solid :dot],
                        xlabel=L"Time \ (ms)",
                        ylabel = "g_Na",
                        dpi=200, 
                        frame_style=:box,
                        grid=false)
                    png(plot_conv_Na_V,       
                        figures_path*"neuron$(i)/plot_gNa_simulated_estimated_itr$(itr)"
                    )

                plot_conv_n_V = plot(ts, [n[:, i].-n0[i] conv_sigma_n_V[:, i]],     
                label=["Original" "Estimated"],
                size=(500,400),
                lc = [:black :red],
                ls = [:solid :dot],
                xlabel=L"Time \ (ms)",
                ylabel = "n",
                dpi=200, 
                frame_style=:box,
                ylims=(-0.01, 0.51),
                grid=false)
                png(plot_conv_n_V,       
                    figures_path*"neuron$(i)/plot_n_simulated_estimated_itr$(itr)"
                )
                plot_conv_n_V = plot(ts, [m[:, i].-m0[i] conv_sigma_m_V[:, i]],     
                label=["Original" "Estimated"],
                size=(500,400),
                lc = [:black :red],
                ls = [:solid :dot],
                xlabel=L"Time \ (ms)",
                ylabel = "m",
                dpi=200, 
                frame_style=:box,
                ylims=(-0.11, 1.01),
                grid=false)
                png(plot_conv_n_V,       
                    figures_path*"neuron$(i)/plot_m_simulated_estimated_itr$(itr)"
                )
                plot_conv_n_V = plot(ts, [h[:, i].-h0[i] conv_sigma_h_V[:, i]],     
                label=["Original" "Estimated"],
                size=(500,400),
                lc = [:black :red],
                ls = [:solid :dot],
                xlabel=L"Time \ (ms)",
                ylabel = "h",
                dpi=200, 
                frame_style=:box,
                ylims=(-1.01, 0.11),
                grid=false)
                png(plot_conv_n_V,       
                    figures_path*"neuron$(i)/plot_h_simulated_estimated_itr$(itr)"
                )#
        end
    end


    ###############################################
    #### NONequilibrium interaction Green Functions

    Eta_s = zeros(resolution, resolution, N, N)
    nonli_pi = zeros(resolution, resolution, N, N)
    g = zeros(resolution, resolution, N, N)


    conv_Eta_s_V = zeros(resolution, N, N) # DeltaS
    for itr=1:6  #### iterative method to approximate DeltaS
        conv_gamma_V_i = zeros(resolution, N)
        for j=1:N
            for i=1:N
                for t=1:resolution
                    for t′=1:(t)   
                        if DeltaV[t′, j] == 0
                            Eta_s[t, t′, i, j]  = 0
                        else
                            # Compute the difference in alpha_S
                            delta_alpha = alpha_S(V[t′, j], a_r[i,j], beta[i, j], V_th[i, j]) - alpha_S(V0[j], a_r[i,j], beta[i, j], V_th[i, j])
                            
                            Eta_s[t, t′, i, j] =  (sigma0[t, t′, i, j]/d_alpha_S(V0[j], a_r[i,j], beta[i, j], V_th[i, j]))*(delta_alpha/DeltaV[t′, j])*(1 - conv_Eta_s_V[t′, i, j]/(1-gS0[i, j]))
                            # exp(-(ts[t]-ts[t′])*(a_d[i,j]+alpha_S(V0[j], a_r[i,j], beta[i,j], V_th[i,j])))*(delta_alpha/DeltaV[t′, j])*(1 -  conv_Eta_s_V[t′, i, j])
                            
                        end
                    end
                end
                
                for t=1:resolution
                    conv_Eta_s_V[t, i, j] = conv(Eta_s[t, 1:t, i, j], DeltaV[1:t, j], dt)
                end
                        
            for t=1:resolution
                for t′=1:(t) 
                    nonli_pi[t, t′, i, j] = conv(gs0[t, t′:t, i, j],  (1 -DeltaV[t′, i]/(Es[i,j].-V0[i]))*Eta_s[t′:t, t′, i, j], dt)
                    g[t, t′, i, j] = gg0[t, t′, i, j] + nonli_pi[t, t′, i, j]
                end

                conv_gamma_V_i[t, i] += conv(g[t, 1:t, i, j], DeltaV[1:t, j], dt)
                
            end
               #=     
                png(
                    heatmap(ts, ts, Eta_s[:, :, i, j],
                            xflip=false,
                            ylabel="t (current time)",
                            xlabel = "t′ (past time)",
                            size=(500,400),
                            title="sigma_$(i),$(j)(t,t′)",
                            fillcolor=:jet1,
                            dpi=200, 
                            frame_style=:box,
                            grid=false),
                            figures_path*"neuron$(i)/Eta_s_ij$(i)_$(j)"
                    )

                png(
                    heatmap(ts, ts, g[:, :, i, j],
                            xflip=false,
                            ylabel="t (current time)",
                            xlabel = "t′ (past time)",
                            title="g_$(i),$(j)(t,t′)",
                            size=(500,400),
                            fillcolor=:jet1,
                            dpi=200, 
                            frame_style=:box,
                            grid=false),
                            figures_path*"neuron$(i)/g_ij$(i)_$(j)"
                    )

                png(
                    heatmap(ts, ts, nonli_pi[:, :, i, j],
                            xflip=false,
                            ylabel="t (current time)",
                            xlabel = "t′ (past time)",
                            title="nonli_pi_$(i),$(j)(t,t′)",
                            size=(500,400),
                            fillcolor=:jet1,
                            dpi=200, 
                            frame_style=:box,
                            grid=false),
                            figures_path*"neuron$(i)/nonli_pi_ij$(i)_$(j)"
                    )
                =#
            end
        end
        if itr%2==0 || itr==1
            plot_Delta_S_sigma_V = plot(ts, [DeltagS[:, 3, 2].+ gS0[3, 2]],     # plot only the synapses that matter for the networks (4←3←2←1)
            ylabel=["Deltagˢ32"],
            label="Simulated",
            size=(500,400),
            lc = :black,
            xlabel="time (ms)",
            dpi=200, 
            frame_style=:box,
            grid=false)
            plot!(ts, [conv_Eta_s_V[:, 3, 2].+ gS0[3, 2]],     # plot only the synapses that matter for the networks (4←3←2←1)
            label="ηˢ32 ∗ DeltaV₁",
            lc = :red,
            ls=:dot,
            lw=2,
            size=(500,400),
            xlabel="time (ms)",
            dpi=200, 
            frame_style=:box,
            grid=false),
            png(plot_Delta_S_sigma_V,       
                figures_path*"/synapses_duration_stim_itr$(itr)"
            )

        end
    end
#=
    plot_Delta_S_sigma_V = plot(ts, [DeltagS[:, 2, 1]],     # plot only the synapses that matter for the networks (4←3←2←1)
        ylabel=["Deltagˢ₂₁"],
        label="Simulated",
        size=(500,400),
        lc = :black,
        xlabel="time (ms)",
        dpi=200, 
        frame_style=:box,
        grid=false)
        plot!(ts, [conv_Eta_s_V[:, 2, 1]],     # plot only the synapses that matter for the networks (4←3←2←1)
        label="ηˢ₂₁ ∗ DeltaV₁",
        lc = :red,
        ls=:dot,
        lw=2,
        size=(500,400),
        xlabel="time (ms)",
        dpi=200, 
        frame_style=:box,
        grid=false),
        png(plot_Delta_S_sigma_V,       
            figures_path*"/synapses_duration_stim"
        )
=#
    #######################################
    ##### Interaction Paths Green functions
    #####    
    # The connected Green function is the directed plus the convolution with the paths from node j to node i
    # First compute the trivial ones, the first neighbors of the boundary condition DeltaVⱼ, then propagate to higher neighbors 
    Gamma0 = g0              # First neighbors       
    Gamma = g                # First neighbors      

    gamma_ast_chi = zeros(size(g))                # First neighbors         # ONLY WORKS FOR CHAIN OF 3 NEURONS
    
    for t=1:resolution
        for t′=1:(t)         # Second neighbors
            Gamma0[t, t′, 3, 1] += conv(g0[t, t′:t, 3, 2], Gamma0[t′:t, t′, 2, 1], dt)
            Gamma[t, t′, 3, 1] += conv(g[t, t′:t, 3, 2], Gamma[t′:t, t′, 2, 1], dt)
        end
    end



    conv_Gamma_V = zeros(resolution, N)
    for i=1:N
        for j=1:N
            if j!=i
                for t=1:resolution
                    conv_Gamma_V[t, i] += conv(Gamma[t, 1:t, i, j], DeltaV[1:t, j], dt)
                    for t′=1:t
                        gamma_ast_chi[t, t′, i, j] = conv(Gamma[t, t′:t, i, j], Chi_i[t′:t, t′, j], dt)

                    end
                end
            end
        end
    end

    est_V = zeros(size(DeltaV))
    for i=1:N
        signal_components[:, i, i] = conv_chi_V_i[:, i]
        ext_comp[:, i] = conv_exptau_I[:, i]
        for j = 1:N
            if j==i 
                continue
            end
            for t=1:resolution
                signal_components[t, i, j] = conv(gamma_ast_chi[t, 1:t, i, j], DeltaV[1:t, j], dt)
            end


        end
        for t=1:resolution
            est_V[t, i] = V0[i] + sum(signal_components[t, i,:]) + ext_comp[t, i]
        end
    end

    neuron_colors= [:green, :blue, :orange]
    input_colors= [:black, :gray, :magenta]

    comp_names = [L"\chi _1 ∗ ΔV _1"      L"G_{1,2} * \chi _2 ∗ ΔV_2"  L"G_{1,3} * \chi _3 ∗ ΔV_3";
                  L"G_{2,1} * \chi _1 ∗ ΔV_1" L"\chi _2 ∗ ΔV _2"       L"G_{2,3} * \chi _3 ∗ ΔV_3";
                  L"G_{3,1} * \chi _1 ∗ ΔV_1" L"G_{3,2} * \chi _2 ∗ ΔV_2"  L"\chi _3 ∗ ΔV _3"]

    plot_signal_components = plot(layout=(N,1), size=(700, 200*N+30), dpi=200, frame_style=:box, grid=false)
    for ni in 1:N

        plot!(plot_signal_components, ts, est_V[:, ni].-V0[ni], lc=neuron_colors[ni], lw=5,la=0.4, label=L"ΔV _{%$ni}", subplot=ni, xlabel=(ni==N ? "Time (ms)" : ""), ylabel="Neuron $(ni)")

        if ni==3
            for nc in 1:N
                if nc>=ni
                    continue
                end
                plot!(plot_signal_components, ts, signal_components[:, ni, nc], lc=neuron_colors[nc], ls=:dot, label=comp_names[ni, nc], subplot=ni)
            end
        end
        #if ni!=3
        #    plot!(plot_signal_components, ts, ext_comp[:, ni], lc=input_colors[ni], ls=:solid, label=L"I(t)", subplot=ni)
        #end
    end



    png(plot_signal_components, figures_path*"DeltaVs_with_signal_components")




    neuron_colors= [:green, :blue, :orange]

    comp_names = [L"\chi _1 ∗ ΔV _1"      L"G_{1,2} * \chi _2 ∗ ΔV_2"  L"G_{1,3} * \chi _3 ∗ ΔV_3";
                  L"G_{2,1} * \chi _1 ∗ ΔV_1" L"\chi _2 ∗ ΔV _2"       L"G_{2,3} * \chi _3 ∗ ΔV_3";
                  L"G_{3,1} * \chi _1 ∗ ΔV_1" L"G_{3,2} * \chi _2 ∗ ΔV_2"  L"\chi _3 ∗ ΔV _3"]

    plot_V_chi_signal_components = plot(layout=(N,1), size=(700, 200*N+30), dpi=200, frame_style=:box, grid=false)
    for ni in 1:N
        plot!(plot_V_chi_signal_components, ts, est_V[:, ni].-V0[ni] .-signal_components[:, ni, ni], lc=neuron_colors[ni], lw=5,la=0.4, label=L"ΔV _{%$ni}-\chi _{%$ni} ∗ ΔV _{%$ni}", subplot=ni, xlabel=(ni==N ? "Time (ms)" : ""), ylabel="Neuron $(ni)")

        if ni==3
            for nc in 1:N
                if nc>=ni
                    continue
                end
                plot!(plot_V_chi_signal_components, ts, signal_components[:, ni, nc], lc=neuron_colors[nc], ls=:dot, label=comp_names[ni, nc], subplot=ni)
            end
        end
        #if ni!=3
        #    plot!(plot_V_chi_signal_components, ts, ext_comp[:, ni], lc=:red, ls=:solid, label=L"G_{%$ni, 1}*(e^{-(t-t')/\tau _0} \ast I(t'))(t)", subplot=ni)
        #end
    end


    png(plot_V_chi_signal_components, figures_path*"DeltaVs-chi_i_signal_components")





    plot_est_V = plot(layout=(N, 1), size=(500, 200*N+20), dpi=200, frame_style=:box, grid=false)
    for i = 1:N
        plot!(plot_est_V, ts, V[:, i], label="Simulated Neuron $(i)", lc=:black, subplot=i, ylabel="V", xlabel=(i==N ? "time (ms)" : ""))
        plot!(plot_est_V, ts, est_V[:, i], label="Estimated", lc=:red, ls=:dot, lw=2, subplot=i)
    end
    png(plot_est_V, figures_path*"/plot_est_V)")



    for j=1:N
        for i=1:N
            if i!=j
                

                png(
                    heatmap(ts, ts, Gamma0[:, :, i, j],
                        xflip=false,
                        ylabel="t (current time)",
                        xlabel = "t′ (past time)",
                        size=(500,400),
                        title="Gamma0$(i),$(j)(t,t′)",
                        fillcolor=:jet1,
                        dpi=200, 
                        frame_style=:box,
                        grid=false),
                        figures_path*"neuron$(i)/Gamma($(N))0_ij$(i)_$(j)"
                    )
                png(
                    heatmap(ts, ts, Gamma[:, :, i, j],
                        xflip=false,
                        ylabel="t (current time)",
                        xlabel = "t′ (past time)",
                        title="Gamma$(i),$(j)(t,t′)",
                        size=(500,500),
                        fillcolor=:jet1,
                        dpi=200, 
                        frame_style=:box,
                        grid=false),
                        figures_path*"neuron$(i)/Gamma($(N))_ij$(i)_$(j)"
                    )
                    png(
                        heatmap(ts, ts, gamma_ast_chi[:, :, i, j],
                            xflip=false,
                            ylabel="t (current time)",
                            xlabel = "t′ (past time)",
                            title="Gamma$(i),$(j)(t,t′)",
                            size=(500,500),
                            fillcolor=:jet1,
                            dpi=200, 
                            frame_style=:box,
                            grid=false),
                            figures_path*"neuron$(i)/Gamma_ast_CHI($(N))_ij$(i)_$(j)"
                        )    

                    plote_level_curves_Gamma = plot(ts[end].-ts[2:end], Gamma0[end, 2:end, i, j],
                                    xlabel="t-t′",
                                    ylabel = "Γᵢⱼ(t,t′)",
                                    label = "Γ₀ᵢⱼ",
                                    lc=:black,
                                    lw=2.4,
                                    size=(800,400),
                                    dpi=200, 
                                    xlims = (0,10),
                                    frame_style=:box,
                                    grid=false)
                    colors_codes = create_rgba_code_line(length(ts_plot))
                    for t_idx in 1:length(ts_plot)
                        plot!(ts[ts_plot[t_idx]].-ts[ts_plot[1]:(ts_plot[t_idx])], Gamma[ts_plot[t_idx], ts_plot[1]:(ts_plot[t_idx]), i, j],
                                    xlabel="t-t′ (ms)",
                                    ylabel = "Γᵢⱼ(t,t′)",
                                    lc=RGBA(colors_codes[t_idx, 1], colors_codes[t_idx, 2], colors_codes[t_idx, 3], colors_codes[t_idx, 4]),
                                    la=0.9,
                                    lw=1.4,
                                    ls= :solid, #rand([:dot; :dot]),
                                    label= "t = "*string(round(ts[ts_plot[t_idx]], digits=1)),
                                    size=(600,400),
                                    dpi=200, 
                                    frame_style=:box,
                                    grid=false)
                            png(plote_level_curves_Gamma,
                                    figures_path*"neuron$(i)/Gamma($(N))_level_curves_ij$(i)_$(j)"
                                )
                    end

    #=
    ####################################################################### plots though only for 3 neurons setup

    plote_level_curves_Gamma = plot(V[1:end, 2], Gamma0[end, 1:end, i, j],
                                xlabel="Neuron 2 State",
                                ylabel = "Gᵢⱼ",
                                label = "G₀",
                                lc=:black,
                                lw=2.4,
                                size=(900,400),
                                dpi=200,
                                legend=:outerright,
                                frame_style=:box,
                                grid=false)

                colors_codes = create_rgba_code_line(length(ts_plot))
                for t_idx in 1:length(ts_plot)
                    plot!(DeltaV[1:ts_plot[t_idx], 2], Gamma[ts_plot[t_idx], 1:(ts_plot[t_idx]), i, j],
                                xlabel="Neuron 2 State",
                                ylabel = "Gᵢⱼ",
                                lc=RGBA(colors_codes[t_idx, 1], colors_codes[t_idx, 2], colors_codes[t_idx, 3], colors_codes[t_idx, 4]),
                                lw=1.4,
                                ls= :solid, #rand([:dot; :dot]),
                                label= "t = "*string(round(ts[ts_plot[t_idx]], digits=1)),
                                size=(900,400),
                                dpi=200,
                                legend=:outerright,
                                frame_style=:box,
                                grid=false)
                        png(plote_level_curves_Gamma,
                                figures_path*"neuron$(i)/Gamma_FUNC_NEURON_2($(N))_level_curves_ij$(i)_$(j)"
                            )
                end

    #######################################################################
    =#
                plote_level_curves_Gamma = plot(ts[1:end], Gamma0[end, 1:end, i, j],
                                xlabel="t′",
                                ylabel = "Γᵢⱼ(t,t′)",
                                label = "Γ₀ᵢⱼ(t,t′)",
                                lc=:black,
                                lw=2.4,
                                size=(900,400),
                                dpi=200,
                                legend=:outerright,
                                frame_style=:box,
                                grid=false)

                vline!([4; 5],
                        lw=1,
                        la=1.0,
                        label="",
                        lc=[:black],
                        ls=[:dot],
                        #annotation=(1.6, 1, text("Stimulus", 12, :black))
                        )

                colors_codes = create_rgba_code_line(length(ts_plot))
                for t_idx in 1:length(ts_plot)
                    plot!(ts[1:(ts_plot[t_idx])], Gamma[ts_plot[t_idx], 1:(ts_plot[t_idx]), i, j],
                                xlabel="t′ (ms)",
                                ylabel = "Γᵢⱼ(t,t′)",
                                lc=RGBA(colors_codes[t_idx, 1], colors_codes[t_idx, 2], colors_codes[t_idx, 3], colors_codes[t_idx, 4]),
                                lw=1.4,
                                ls= :solid, #rand([:dot; :dot]),
                                label= "t = "*string(round(ts[ts_plot[t_idx]], digits=1)),
                                size=(900,400),
                                dpi=200,
                                legend=:outerright,
                                frame_style=:box,
                                grid=false)
                        png(plote_level_curves_Gamma,
                                figures_path*"neuron$(i)/Gamma($(N))_level_curves_ij$(i)_$(j)_no_translation"
                            )
                end

                ts_gif_plot = 1:10:resolution  # Time indices to plot (adjust according to your data)
                colors_codes = create_rgba_code_line(length(ts_plot))

                #= Create the animation object
                anim = @animate for t_idx in ProgressBar(1:2:resolution)

                    # Base plot (level curves, black line)
                    plote_level_curves_Gamma_gif = plot(ts[1:t_idx], Gamma0[t_idx, 1:t_idx, i, j],
                                            xlabel="t′ (ms)",
                                            ylabel = "Γ _$(i),$(j)",
                                            label = "Γ₀ _$(i),$(j)",
                                            ylim = (-0.1, 1.0),
                                            lc=:gray,
                                            la=0.8,
                                            lw=3.0,
                                            size=(900,400),
                                            dpi=200,
                                            frame_style=:box,
                                            title = "Time = $(round(ts[t_idx], digits=1)) ms",
                                            grid=false)

                                        plot!([0; 2.0; 2.0; 6.0; 6.0; 20.0;20.0; 24.0;24.0; tf], 
                                            [-1.0e-2;-1.0e-2;1.0e-1;1.0e-1; -1.0e-2; -1.0e-2; 1.0e-1; 1.0e-1;-1.0e-2;-1.0e-2]*maximum(Gamma[:, :, i, j]), label ="", lc=:red)
                    # Add plot data for the current frame (up to the current t_idx)
                    plot!(ts[1:t_idx], Gamma[t_idx, 1:t_idx, i, j],
                        #xlabel="t′",
                        ylabel = "Γ _$(i),$(j)",
                        lc=:Black,#RGBA(colors_codes[t_idx, 1], colors_codes[t_idx, 2], colors_codes[t_idx, 3], colors_codes[t_idx, 4]),
                        lw=1.5,
                        ls=:solid,
                        size=(900,400),
                        label="Γ _$(i),$(j)($(round(ts[t_idx], digits=1)), t′)",
                        dpi=200,
                        frame_style=:box,
                        grid=false)
                end

                # Save the animation as a gif
                gif(anim, figures_path*"neuron$(i)/Gamma_GIF_ij$(i)_$(j)_no_translation.gif", fps=20)  # Adjust fps as needed
                =#
                plote_level_curves_Gamma_ast_Chi = plot(
                                size=(800,400),
                                dpi=200, 
                                xlims = (0,10),
                                ylims = (-0.1,2.0),
                                frame_style=:box,
                                grid=false)
                colors_codes = create_rgba_code_line(length(ts_plot))
                for t_idx in 1:length(ts_plot)
                    plot!(ts[ts_plot[t_idx]].-ts[ts_plot[1]:(ts_plot[t_idx])], gamma_ast_chi[ts_plot[t_idx], ts_plot[1]:(ts_plot[t_idx]), i, j],
                                xlabel="t-t′ (ms)",
                                ylabel = L"\gamma _{ik} \ast \chi_k",
                                lc=RGBA(colors_codes[t_idx, 1], colors_codes[t_idx, 2], colors_codes[t_idx, 3], colors_codes[t_idx, 4]),
                                la=0.9,
                                lw=1.4,
                                ls= :solid, #rand([:dot; :dot]),
                                label= "t = "*string(round(ts[ts_plot[t_idx]], digits=1)),
                                size=(600,400),
                                dpi=200, 
                                frame_style=:box,
                                grid=false)
                        png(plote_level_curves_Gamma_ast_Chi,
                                figures_path*"neuron$(i)/gamma_ast_chi($(N))_level_curves_ij$(i)_$(j)"
                            )
                end

                plote_level_curves_Gamma_ast_Chi = plot(
                                size=(900,400),
                                dpi=200,
                                legend=:outerright,
                                frame_style=:box,
                                grid=false)

                vline!([4; 5],
                        lw=1,
                        la=1.0,
                        label="",
                        lc=[:black],
                        ls=[:dot],
                        #annotation=(1.6, 1, text("Stimulus", 12, :black))
                        )

                colors_codes = create_rgba_code_line(length(ts_plot))
                for t_idx in 1:length(ts_plot)
                    plot!(ts[1:(ts_plot[t_idx])], gamma_ast_chi[ts_plot[t_idx], 1:(ts_plot[t_idx]), i, j],
                                xlabel="t′ (ms)",
                                ylabel =  L"\gamma _{ik} \ast \chi_k",
                                lc=RGBA(colors_codes[t_idx, 1], colors_codes[t_idx, 2], colors_codes[t_idx, 3], colors_codes[t_idx, 4]),
                                lw=1.4,
                                ls= :solid, #rand([:dot; :dot]),
                                label= "t = "*string(round(ts[ts_plot[t_idx]], digits=1)),
                                size=(900,400),
                                dpi=200,
                                legend=:outerright,
                                frame_style=:box,
                                grid=false)
                        png(plote_level_curves_Gamma_ast_Chi,
                                figures_path*"neuron$(i)/Gamma_ast_CHI($(N))_level_curves_ij$(i)_$(j)_no_translation"
                            )
                end


                plote_level_curves_Gamma_minusEQ= plot()

                vline!([4; 5],
                        lw=1,
                        la=1.0,
                        label="",
                        lc=[:black],
                        ls=[:dot],
                        #annotation=(1.6, 0.5, text("Stimulus", 12, :black))
                        )

                colors_codes = create_rgba_code_line(length(ts_plot))
                for t_idx in 1:length(ts_plot)
                    plot!(ts[1:(ts_plot[t_idx])], Gamma[ts_plot[t_idx], 1:(ts_plot[t_idx]), i, j].-Gamma0[ts_plot[t_idx], 1:(ts_plot[t_idx]), i, j],
                                xlabel="t′ (ms)",
                                ylabel = L"γ_{i,j}-γ_{0,ij}",
                                lc=RGBA(colors_codes[t_idx, 1], colors_codes[t_idx, 2], colors_codes[t_idx, 3], colors_codes[t_idx, 4]),
                                lw=1.4,
                                ls= :solid, #rand([:dot; :dot]),
                                label= "t = "*string(round(ts[ts_plot[t_idx]], digits=1)),
                                size=(900,400),
                                dpi=200, 
                                frame_style=:box,
                                legend=:outerright,
                                grid=false)
                        png(plote_level_curves_Gamma_minusEQ,
                                figures_path*"neuron$(i)/Gamma($(N))_level_curves_ij$(i)_$(j)_no_translation_minus_eq"
                            )
                end

            end
        end
    end
    
end

main()
