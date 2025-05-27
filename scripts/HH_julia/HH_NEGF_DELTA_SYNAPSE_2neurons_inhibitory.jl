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

pythonplot()

function Θ(x)
    if x<0
        return 0.0
    else
        return 1.0
    end
end

function alpha_S(Veq, beta, V_th)
    return 1/(1 + exp(-beta*(Veq-V_th)))
end
function d_alpha_S(Veq, beta, V_th)
    return ((beta*exp(-beta*(Veq-V_th)))/(1 + exp(-beta*(Veq-V_th)))^2)
end


function conv(X, Y, interval)
    return ((interval[end]-interval[1])/length(interval))*sum(skipmissing(X.*Y))
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
    C_m, g_Na_bar, g_K_bar, g_L_bar, E_Na, E_K, E_L, N, A, B, Es, a_r, a_d, beta, V_th, I_ext, stim = p

    for i=1:N # Iteration over neurons variables
        V, m, h, n = u[i, 1:4]

        I_ext = 0.0
        if i == 1 && (t>=4.0 && t<=5.0) # || (t>=10.0 && t<=11.0) || (t>=20.0 && t<=21.0) # Adds external stimulation
            I_ext += stim
        end

        for j=1:N # Coupling terms from pre-synaptic neurons
            I_ext += -A[i, j]*(u[i,1] - u[j,1]) - B[i,j]*u[i,4+j]*(u[i, 1] - Es[i,j])
            du[i, 4+j] = (a_r[i,j]*(1/(1 + exp(-beta[i,j]*(u[j,1]-V_th[i,j]))))*(1-u[i, 4+j])-a_d[i,j]*u[i, 4+j])
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
    tf =  50.0
    
    resolution = 1000 

    tspan = (ti, tf)

    tₛ = range(ti, tf, length=resolution)

    # Neuron Parameters
    # Define constants
    C_m      =   1.0         # membrane capacitance, in uF/cm^2
    g_Na_bar = 120.0         # maximum conductances, in mS/cm^2
    g_K_bar  =  36.0
    g_L_bar  =   0.5
    E_Na     =  50.0         # Nernst reversal potentials, in mV
    E_K      = -77.0
    E_L      = -55.0 

    # Synapses parameters
    N    = 2 # Number of neurons
    
    # Coupling matrices, adjacency matrices
    A    = [  0.0  0.0;  # S/F   # Gap junction coupling  
            0.0  0.0]   

    B    = [  0.0  0.0;  # S/F    # Chemical synapse coupling
             0.5  0.0] 

    Es   = fill(0.0, (N,N))        # mV  # Synapse Nerst potential, determines excitation or inhibition 
    Es[2, 1] = -70                # Change this to control Inhibition(-70 value) depends on the synaptic receptor ion potential
    a_r  = fill(5.0, (N,N))        # rates for synaptic channel conductance activity  
    a_d  = fill(5.0, (N,N))
    #a_r[4, 1] = 1.0
    beta    = fill(0.125, (N,N))      # (mV)⁻¹
    V_th = fill(-50.0  , (N,N))    #  mV    # update when having the equilibrium values
    #V_th[3, 2] = -10.0             #  mV    # alpha_S(V=V_th) = 1/2

    # External current
    I_ext = 0.0  # in μA/cm^2     
    stim  = 8.0  # in μA/cm^2

                
    ts_plot = [50; 80; 100; 120; 150; 180; 200; 220; 250; 280; 300; 350; 400; 450; 500; 550; 600; 700; 800]

    figures_path = "/home/gabriel/figuras/qualification/NEGF_HH_DELTA_SYNAPSE_$(N)neurons_inhibitory/$(g_Na_bar)_$(g_K_bar)_$(g_L_bar)_$(E_Na)_$(E_K)_$(E_L)_syn_$(minimum(Es))_stim_$(I_ext)_$(stim)/"
    
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
        prob = ODEProblem(hh_model!, u0, tspan, [C_m, g_Na_bar, g_K_bar, g_L_bar, E_Na, E_K, E_L, N, A, B, Es, a_r, a_d, beta, V_th, I_ext, 0.0])
        sol = solve(prob, Tsit5(),  dt=0.001, saveat=tₛ, reltol=1e-9, abstol=1e-9, maxiters = 1e7)

    #    png(plot(sol, layout=(4,1), lc=:black, xlabel=["" "" "" "Time (ms)"], ylabel=["V(t) (mV)" "m(t)" "h(t)" "n(t)"], label="", frame_style=:box, size=(500,500), dpi=200), figures_path*"neuron$(i)/HH_variables_simulation_transient")

        u0 = sol[end]
        
        for i=1:N
            V0[i], m0[i], h0[i], n0[i] = u0[i,1:4]
        end
        
        gS0 = u0[:, 5:end]

        V_th = [fill(V0[1], N) fill(V0[2], N)] 
    #    V_th[3, 2] = -10.0             #  mV    # alpha_S(V=V_th) = 1/2
    #    V_th[4, 1] = -10.0             #  mV    # alpha_S(V=V_th) = 1/2
    end

    # Solve the differential equations STIMULATED
    prob = ODEProblem(hh_model!, u0, tspan, [C_m, g_Na_bar, g_K_bar, g_L_bar, E_Na, E_K, E_L, N, A, B, Es, a_r, a_d, beta, V_th, I_ext, stim])
    sol = solve(prob, Tsit5(),  dt=0.001, saveat=tₛ, reltol=1e-9, abstol=1e-9, maxiters = 1e7)

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
        tau_i[i] = (C_m)./(g_L_bar.+(g_Na_bar*m0[i]^(3)*h0[i]).+(g_K_bar*n0[i]^4) + sum(A*gS0) + sum(B*ones(N)))
    end

    plot_Vs =  plot(sol.t, V, layout=(N,1), 
    lc=:black, xlabel=["" "Time (ms)"], ylabel="V(t)", label="", 
    frame_style=:box, size=(500, 200*N+20), dpi=200, grid=false)
    png(plot_Vs, figures_path*"Vs_variables_simulation_neurons")
    
    plot_Delta_Vs =  plot(sol.t, DeltaV, layout=(N,1), 
    lc=:black, xlabel=["" L"Time (ms)"], ylabel=L"\Delta V(t)", label="", 
    frame_style=:box, size=(500, 200*N+20), dpi=200, grid=false)
    png(plot_Delta_Vs, figures_path*"DeltaVs_variables_simulation_neurons")


    for i=1:N
        var_plots = plot(sol.t, [V[:, i] m[:, i] h[:, i] n[:, i]], layout=(4,1), 
        lc=:black, xlabel=["" "" "" "Time (ms)"], ylabel=["V(t) (mV)" "m(t)" "h(t)" "n(t)"], label="", 
        frame_style=:box, size=(900,500), dpi=200, grid=false)
        hline!([V0[i] m0[i] h0[i] n0[i]], label="", lc=:black, ls=:dash)
        
        png(var_plots, figures_path*"neuron$(i)/HH_variables_simulation_neurons")
            
        var_syn_plots = plot(sol.t, [gS[:, i,1] gS[:, i,2]], layout=(4,1), 
            lc=:black, xlabel=["" "Time (ms)"], ylabel=["gS$(i)1" "gS$(i)2"], label="", 
            frame_style=:box, size=(900,500), dpi=200, grid=false)
            hline!([gS0[i,1] gS0[i,2]], label="", lc=:black, ls=:dash)
        
        png(var_syn_plots,figures_path*"neuron$(i)/Synaptic_var_gS_post_syn")


        cond_plote = plot(tₛ, [V[:, i] (g_Na_bar*m[:, i].^(3).*h[:, i]) (g_K_bar*n[:, i].^4)], layout=(3,1), 
        lc=:black, xlabel=["" "" "Time (ms)"], ylabel=["V(t)(mV)" "g_{Na} (t)" "g_{K} (t)"], label="", 
        frame_style=:box, size=(900,500), dpi=200, left_margin=5mm, grid=false)
        hline!([V0[i] (g_Na_bar*m0[i].^(3).*h0[i]) (g_K_bar*n0[i].^4)], label=["Equilibrium" "" "" ""], lc=:red, ls=:dash)
        
        png(cond_plote, figures_path*"neuron$(i)/HH_conductances_simulation")

        png(plot(tₛ, [(tau_n[:, i]) (tau_m[:, i]) (tau_h[:, i])], layout=(3,1), 
        lc=:black, xlabel=["" "" "Time(ms)"], ylabel=["n" "m" "h"], label="", 
        frame_style=:box, size=(500,500), dpi=200, grid=false),
        figures_path*"neuron$(i)/HH_taus")

        png(plot(tₛ, [(n_inf[:, i]) (m_inf[:, i]) (h_inf[:, i])], layout=(3,1), 
        lc=:black, xlabel=["" "" "Time(ms)"], ylabel=["n" "m" "h"], label="", 
        frame_style=:box, size=(500,500), dpi=200, grid=false),
        figures_path*"neuron$(i)/HH_saturation variables")
    end

    #### Equilibrium effective green functions
    sigma₀  = zeros(resolution, resolution, N, N)
    gg₀ = zeros(resolution, resolution, N, N)
    gs₀ = zeros(resolution, resolution, N, N)
    g₀  = zeros(resolution, resolution, N, N)

    for j=1:N
        for i=j:N
            for t′=1:resolution
                sigma₀[:, t′, i, j]  = Θ.(tₛ.-tₛ[t′]) .*a_r[i,j]*(1-gS0[i, j]).*d_alpha_S(V0[j], beta[i,j], V_th[i,j]).*exp.(-(tₛ.-tₛ[t′]).*(a_d[i,j]-a_r[i,j]/(1 + exp(-beta[i,j]*(V0[j]-V_th[i,j])))))
                gg₀[:, t′, i, j] = Θ.(tₛ.-tₛ[t′]) .*A[i, j].*exp.(-(tₛ.-tₛ[t′])/(tau_i[i]))
                gs₀[:, t′, i, j] = Θ.(tₛ.-tₛ[t′]) .*B[i, j].*(Es[i,j]-V0[i]).*exp.(-(tₛ.-tₛ[t′])/(tau_i[i]))
            end
            png(
                heatmap(tₛ, tₛ, sigma₀[:, :, i, j],
                    xflip=false,
                    ylabel="t (current time)",
                    xlabel = "t′ (past time)",
                    size=(500,400),
                    fillcolor=:jet1,
                    dpi=200, 
                    frame_style=:box,
                    grid=false),
                figures_path*"neuron$(i)/sigmaZERO_ij$(i)_$(j)"
            )

            png(heatmap(tₛ, tₛ, gs₀[:, :, i, j],
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

            png(heatmap(tₛ, tₛ, gg₀[:, :, i, j],
            xflip=false,
            ylabel="t (current time)",
            xlabel = "t′ (past time)",
            size=(500,400),
            fillcolor=:jet1,
            dpi=200, 
            frame_style=:box,
            grid=false),
                figures_path*"neuron$(i)/ggZERO_ij$(i)_$(j)"
            )
        end
    end

    for t=1:resolution
        for t′=1:(t-1)
            for i=1:N
                for j=1:N
                    g₀[t, t′, i, j] = gg₀[t, t′, i, j] + conv(gs₀[t, t′:t, i, j], sigma₀[t′:t, t′, i, j], tₛ[t′:t])
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
    
    Chi_K  = zeros(resolution, resolution, N)
    Chi_Na = zeros(resolution, resolution, N)

    Chi_i  = zeros(resolution, resolution, N)
    conv_chi_V_i = zeros(resolution, N)

    for itr=ProgressBar(1:6)  #### iterative method to approximate sigma
        for i=1:N
            for t=1:resolution
                for t′=1:(t) 
                   
                    if (DeltaV[t′, i] != 0.0)
                        # Potassium Channel
                        sigma_n_V[t, t′, i]  = (n_inf[t′, i].- (n_inf0[i] .+ (conv_sigma_n_V[t′, i]))) / (tau_n[t′, i]*(DeltaV[t′, i]))
                        kappa_gK_V[t, t′, i] = g_K_bar*(conv_sigma_n_V[t′, i]^3)*((n_inf[t′, i]- n_inf0[i]) - (conv_sigma_n_V[t′, i]))/(tau_m[t′, i]*(DeltaV[t′, i]))

                        # Sodium Channel
                        sigma_m_V[t, t′, i]  = (m_inf[t′, i].- (m_inf0[i] .+ (conv_sigma_m_V[t′, i]))) / (tau_m[t′, i]*(DeltaV[t′, i]))
                        sigma_h_V[t, t′, i]  = (h_inf[t′, i].- (h_inf0[i] .+ (conv_sigma_h_V[t′, i]))) / (tau_h[t′, i]*(DeltaV[t′, i]))
                        kappa_gNa_V[t, t′, i] = (g_K_bar*conv_sigma_m_V[t′, i]^2)*(conv_sigma_h_V[t′, i]*(((m_inf[t′, i]- m_inf0[i]) - conv_sigma_m_V[t′, i])/(tau_m[t′, i]*(DeltaV[t′, i]))) + conv_sigma_m_V[t′, i]*(((h_inf[t′, i]- h_inf0[i]) - conv_sigma_h_V[t′, i])/(tau_h[t′, i]*(DeltaV[t′, i]))))
                    end
                    Chi_K[t, t′, i]  = -exp(-(t-t′)/tau_i[i])*(V[t′, i] - E_K )/C_m
                    Chi_Na[t, t′, i] = -exp(-(t-t′)/tau_i[i])*(V[t′, i] - E_Na)/C_m
                end
                conv_sigma_n_V[t, i] = conv(sigma_n_V[t, 1:t, i], DeltaV[1:t, i], tₛ[1:t])
                conv_sigma_m_V[t, i] = conv(sigma_m_V[t, 1:t, i], DeltaV[1:t, i], tₛ[1:t])
                conv_sigma_h_V[t, i] = conv(sigma_h_V[t, 1:t, i], DeltaV[1:t, i], tₛ[1:t])
                
            end
            for t=1:resolution
                for t′=1:t
                    Chi_i[t, t′, i] = conv(Chi_K[t, t′:t, i], kappa_gK_V[t′:t, t′, i], tₛ[t′:t]) + conv(Chi_Na[t, t′:t, i], kappa_gNa_V[t′:t, t′, i], tₛ[t′:t])
                end
                conv_chi_V_i[t, i] = conv(Chi_i[t, 1:t, i], DeltaV[1:t, i], tₛ[1:t])
            end

            plot_Vs_minus_conv_chi_V_i =  plot(tₛ, DeltaV .- conv_chi_V_i, layout=(N,1), 
            lc=:black, xlabel=["" L"Time (ms)"], ylabel=L"\Delta V_i(t) - (\chi_i \ast \Delta V_i)(t)", label=["Neuron 1" "Neuron 2"], 
            frame_style=:box, size=(500, 200*N+20), dpi=200, grid=false)

            png(plot_Vs_minus_conv_chi_V_i, figures_path*"DeltaVs_minus_conv_ChiVs_variables_simulation_neurons")

            display(conv_chi_V_i)

                png(
                    heatmap(tₛ, tₛ, Chi_i[:, :, i],
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

                plote = plot(tₛ[ts_plot[1]:(end)], Chi_i[end, ts_plot[1]:(end), i],
                    ylabel= L"χᵢ(t,t′)",
                    label="t=$(round(tₛ[end],digits=1))",
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
                display(colors_codes)
                for t_idx in 1:length(ts_plot)
                    plot!(tₛ[1:(ts_plot[t_idx])], Chi_i[ts_plot[t_idx], 1:ts_plot[t_idx], i],
                        ylabel= L"χᵢ(t,t′)",
                        label="t=$(round(tₛ[ts_plot[t_idx]],digits=1))",
                        xlabel = "t′ (ms)",
                        lc=RGBA(colors_codes[t_idx, 1], colors_codes[t_idx, 2], colors_codes[t_idx, 3], colors_codes[t_idx, 4]),
                        ls=:solid,
                        lw=1.4,
                        size=(900,400),
                        dpi=200,
                        legend=:outerright,
                        frame_style=:box,
                        #xlims=(0,tₛ[end]),
                        #left_margin=2mm,
                        grid=false)
                end

                png(plote,figures_path*"neuron$(i)/level_CHI_i")

                plote = plot(tₛ[ts_plot[1]:(end)], Chi_i[end, ts_plot[1]:(end), i],
                    ylabel= L"χᵢ(t,t′)",
                    label="t=$(round(tₛ[end],digits=1))",
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
                display(colors_codes)
                for t_idx in 1:length(ts_plot)
                    plot!(tₛ[1:(ts_plot[t_idx])], Chi_i[ts_plot[t_idx], 1:ts_plot[t_idx], i],
                        ylabel= L"χᵢ(t,t′)",
                        label="t=$(round(tₛ[ts_plot[t_idx]],digits=1))",
                        xlabel = "t′ (ms)",
                        lc=RGBA(colors_codes[t_idx, 1], colors_codes[t_idx, 2], colors_codes[t_idx, 3], colors_codes[t_idx, 4]),
                        ls=:solid,
                        lw=1.4,
                        size=(900,400),
                        dpi=200,
                        legend=:outerright,
                        frame_style=:box,
                        #xlims=(0,tₛ[end]),
                        #left_margin=2mm,
                        grid=false)
                end
                
                png(plote,figures_path*"neuron$(i)/level_CHI_i_zoomIN")

                png(
                    heatmap(tₛ, tₛ, sigma_n_V[:, :, i],
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
                        heatmap(tₛ, tₛ, kappa_gK_V[:, :, i],
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
                    heatmap(tₛ, tₛ, sigma_m_V[:, :, i],
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
                    heatmap(tₛ, tₛ, sigma_h_V[:, :, i],
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
                    plot(tₛ[end] .- tₛ, sigma_n_V[end, 1:end, i],
                            xflip=false,
                            ylabel= ["sigma_{n,V}(t,t′)"],
                            label="",
                            xlabel = "t - t′",
                            title="t=$(tₛ[end])",
                            size=(1000,800),
                            lc=[:black :red],
                            ls=[:solid :dash],
                            dpi=200, 
                            frame_style=:box,
                            #ylims = (-0.1,0.1),
                            grid=false),
                            figures_path*"neuron$(i)/level_curve_sigma_n_V"
                    )
                    png(
                        plot(tₛ[end] .- tₛ, [kappa_gK_V[end, 1:end, i] ],
                                xflip=false,
                                ylabel= ["kappa_{K,V}(t,t′)"],
                                label="",
                                xlabel = "t - t′",
                                title="t=$(tₛ[end])",
                                size=(1000,800),
                                lc=[:black :red],
                                ls=[:solid :dash],
                                dpi=200, 
                                frame_style=:box,
                                #ylims = (-0.1,0.1),
                                grid=false),
                                figures_path*"neuron$(i)/level_curve_Gamma_K_V"
                        )
                        png(
                            plot(tₛ[end] .- tₛ, [kappa_gNa_V[end, 1:end, i] ],
                                    xflip=false,
                                    ylabel= ["kappa_{Na,V}(t,t′)"],
                                    label="",
                                    xlabel = "t - t′",
                                    title="t=$(tₛ[end])",
                                    size=(1000,800),
                                    lc=[:black :red],
                                    ls=[:solid :dash],
                                    dpi=200, 
                                    frame_style=:box,
                                    #ylims = (-0.1,0.1),
                                    grid=false),
                                    figures_path*"neuron$(i)/level_curve_Gamma_Na_V"
                            )
                    png(
                        plot(tₛ, [sigma_m_V[end, 1:end, i] ],
                                xflip=false,
                                ylabel= ["sigma_{m,V}(t,t′)"],
                                label="",
                                xlabel = "t - t′",
                                title="t=$(tₛ[end])",
                                size=(1000,800),
                                lc=[:black :red],
                                ls=[:solid :dash],
                                dpi=200, 
                                frame_style=:box,
                                ylims = (-0.1,0.1),
                                grid=false),
                                figures_path*"neuron$(i)/level_curve_sigma_m_V"
                        )
                        png(
                            plot(tₛ[end] .- tₛ, [sigma_m_V[end, 1:end, i] ],
                                    xflip=false,
                                    ylabel= ["sigma_{h,V}(t,t′)"],
                                    label="",
                                    xlabel = "t - t′",
                                    title="t=$(tₛ[end])",
                                    size=(1000,800),
                                    lc=[:black :red],
                                    ls=[:solid :dash],
                                    dpi=200, 
                                    frame_style=:box,
                                    ylims = (-0.1,0.1),
                                    grid=false),
                                    figures_path*"neuron$(i)/level_curve_sigma_h_V"
                            )
                plot_conv_n_V = plot(tₛ, [n[:, i].-n0[i] conv_sigma_n_V[:, i]],     
                label=["Original" "Estimated"],
                size=(500,400),
                lc = [:black :red],
                ls = [:solid :dash],
                xlabel="Time (ms)",
                ylabel = "n",
                dpi=200, 
                frame_style=:box,
                ylims=(-0.01, 0.51),
                grid=false)
                png(plot_conv_n_V,       
                    figures_path*"neuron$(i)/plot_n_simulated_estimated_itr$(itr)"
                )
                plot_conv_n_V = plot(tₛ, [m[:, i].-m0[i] conv_sigma_m_V[:, i]],     
                label=["Original" "Estimated"],
                size=(500,400),
                lc = [:black :red],
                ls = [:solid :dash],
                xlabel="Time (ms)",
                ylabel = "m",
                dpi=200, 
                frame_style=:box,
                ylims=(-0.11, 1.01),
                grid=false)
                png(plot_conv_n_V,       
                    figures_path*"neuron$(i)/plot_m_simulated_estimated_itr$(itr)"
                )
                plot_conv_n_V = plot(tₛ, [h[:, i].-h0[i] conv_sigma_h_V[:, i]],     
                label=["Original" "Estimated"],
                size=(500,400),
                lc = [:black :red],
                ls = [:solid :dash],
                xlabel="Time (ms)",
                ylabel = "h",
                dpi=200, 
                frame_style=:box,
                ylims=(-1.01, 0.11),
                grid=false)
                png(plot_conv_n_V,       
                    figures_path*"neuron$(i)/plot_h_simulated_estimated_itr$(itr)"
                )
        end
    end


    ###############################################
    #### NONequilibrium interaction Green Functions

    Chi_s = zeros(resolution, resolution, N, N)
    𝜋 = zeros(resolution, resolution, N, N)
    g = zeros(resolution, resolution, N, N)


    conv_Chi_s_V = zeros(resolution, N, N) # DeltaS
    for itr=1:10  #### iterative method to approximate sigma
        for j=1:N
            for i=j:N
                for t=1:resolution
                    for t′=1:(t-1)   
                        if V[t′, j] == V0[j]
                            Chi_s[t, t′, i, j]  = 0.0 #(sigma₀[t, t′, i, j]/d_alpha_S(V0[j], beta[i, j], V_th[i, j]))*0*(1 - (Ss[t′, i, j]-gS0[i, j])/(1-gS0[i, j]))
                        else
                            Chi_s[t, t′, i, j]  = (sigma₀[t, t′, i, j]/d_alpha_S(V0[j], beta[i, j], V_th[i, j]))*((alpha_S(V[t′, j], beta[i, j], V_th[i,j])-alpha_S(V0[j], beta[i, j], V_th[i,j]))/DeltaV[t′, j])*(1 - (conv_Chi_s_V[t′, i, j])/(1-gS0[i, j]))
                        end
                    end
                end
                for t=1:resolution
                    conv_Chi_s_V[t, i, j] = conv(Chi_s[t, 1:t, i, j], DeltaV[1:t, j], tₛ[1:t])
                end
                        
            for t=1:resolution
                for t′=1:(t-1) 
                    𝜋[t, t′, i, j] = conv(gs₀[t, t′:t, i, j], (1 .-(DeltaV[t′:t, i]/(Es[i, j]-V0[i]))).*Chi_s[t′:t, t′, i, j], tₛ[t′:t])
                    g[t, t′, i, j] = gg₀[t, t′, i, j] + 𝜋[t, t′, i, j]
                end
            end
                    
                png(
                    heatmap(tₛ, tₛ, Chi_s[:, :, i, j],
                            xflip=false,
                            ylabel="t (current time)",
                            xlabel = "t′ (past time)",
                            size=(500,400),
                            title="sigma_$(i),$(j)(t,t′)",
                            fillcolor=:jet1,
                            dpi=200, 
                            frame_style=:box,
                            grid=false),
                            figures_path*"neuron$(i)/Chi_s_ij$(i)_$(j)"
                    )

                png(
                    heatmap(tₛ, tₛ, g[:, :, i, j],
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
                    heatmap(tₛ, tₛ, 𝜋[:, :, i, j],
                            xflip=false,
                            ylabel="t (current time)",
                            xlabel = "t′ (past time)",
                            title="𝜋_$(i),$(j)(t,t′)",
                            size=(500,400),
                            fillcolor=:jet1,
                            dpi=200, 
                            frame_style=:box,
                            grid=false),
                            figures_path*"neuron$(i)/𝜋_ij$(i)_$(j)"
                    )
            end
        end
        if itr%5==0 || itr==1
            plot_Delta_S_sigma_V = plot(tₛ, [DeltagS[:, 2, 1]],     # plot only the synapses that matter for the networks (4←3←2←1)
            ylabel=["Deltagˢ₂₁"],
            label="Simulated",
            size=(500,400),
            lc = :black,
            xlabel="time (ms)",
            dpi=200, 
            frame_style=:box,
            grid=false)
            plot!(tₛ, [conv_Chi_s_V[:, 2, 1]],     # plot only the synapses that matter for the networks (4←3←2←1)
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
                figures_path*"/synapses_duration_stim_itr$(itr)"
            )
        end
    end

    plot_Delta_S_sigma_V = plot(tₛ, [DeltagS[:, 2, 1]],     # plot only the synapses that matter for the networks (4←3←2←1)
        ylabel=["Deltagˢ₂₁"],
        label="Simulated",
        size=(500,400),
        lc = :black,
        xlabel="time (ms)",
        dpi=200, 
        frame_style=:box,
        grid=false)
        plot!(tₛ, [conv_Chi_s_V[:, 2, 1]],     # plot only the synapses that matter for the networks (4←3←2←1)
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

    #######################################
    ##### Interaction Paths Green functions
    #####    
    # The connected Green function is the directed plus the convolution with the paths from node j to node i
    # First compute the trivial ones, the first neighbors of the boundary condition DeltaVⱼ, then propagate to higher neighbors 
    Gamma₀ = g₀              # First neighbors       
    Gamma = g                # First neighbors       
    conv_Gamma_V = zeros(resolution, N)
    for i=1:N
        for t=1:resolution
            for j=1:N
                conv_Gamma_V[t, i] += conv(g[t, 1:t, i, j], DeltaV[1:t, j], tₛ[1:t])
            end
        end
    end


    for j=1:N-1
        for i=j:N
            png(
                heatmap(tₛ, tₛ, Gamma₀[:, :, i, j],
                    xflip=false,
                    ylabel="t (current time)",
                    xlabel = "t′ (past time)",
                    size=(500,400),
                    title="Gamma₀$(i),$(j)(t,t′)",
                    fillcolor=:jet1,
                    dpi=200, 
                    frame_style=:box,
                    grid=false),
                    figures_path*"neuron$(i)/Gamma($(N))0_ij$(i)_$(j)"
                )
            png(
                heatmap(tₛ, tₛ, Gamma[:, :, i, j],
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

            plote_level_curves_Gamma = plot(tₛ[end].-tₛ[2:end], Gamma₀[end, 2:end, i, j],
                            xlabel="t-t′",
                            ylabel = "γ",
                            label = "γ₀",
                            lc=:black,
                            lw=2.4,
                            size=(800,400),
                            dpi=200, 
                            xlims = (0,10),
                            ylims = (-0.1,2.0),
                            frame_style=:box,
                            grid=false)
            colors_codes = create_rgba_code_line(length(ts_plot))
            for t_idx in 1:length(ts_plot)
                plot!(tₛ[ts_plot[t_idx]].-tₛ[ts_plot[1]:(ts_plot[t_idx])], Gamma[ts_plot[t_idx], ts_plot[1]:(ts_plot[t_idx]), i, j],
                            xlabel="t-t′ (ms)",
                            ylabel = "γᵢⱼ",
                            lc=RGBA(colors_codes[t_idx, 1], colors_codes[t_idx, 2], colors_codes[t_idx, 3], colors_codes[t_idx, 4]),
                            la=0.9,
                            lw=1.4,
                            ls= :solid, #rand([:dash; :dot]),
                            label= "t = "*string(round(tₛ[ts_plot[t_idx]], digits=2)),
                            size=(600,400),
                            dpi=200, 
                            frame_style=:box,
                            grid=false)
                    png(plote_level_curves_Gamma,
                            figures_path*"neuron$(i)/Gamma($(N))_level_curves_ij$(i)_$(j)"
                        )
            end
            plote_level_curves_Gamma = plot(tₛ[1:end], Gamma₀[end, 1:end, i, j],
                            xlabel="t′",
                            ylabel = "γ",
                            label = "γ₀",
                            lc=:black,
                            lw=2.4,
                            size=(900,400),
                            dpi=200,
                            legend=:outerright,
                            ylims = (-0.1,2.0),
                            frame_style=:box,
                            grid=false)

            vline!([4; 5],
                    lw=1,
                    la=1.0,
                    label="",
                    lc=[:black],
                    ls=[:dot],
                    annotation=(1.6, 1, text("Stimulus", 12, :black))
                    )

            colors_codes = create_rgba_code_line(length(ts_plot))
            for t_idx in 1:length(ts_plot)
                plot!(tₛ[1:(ts_plot[t_idx])], Gamma[ts_plot[t_idx], 1:(ts_plot[t_idx]), i, j],
                            xlabel="t′ (ms)",
                            ylabel = "γᵢⱼ",
                            lc=RGBA(colors_codes[t_idx, 1], colors_codes[t_idx, 2], colors_codes[t_idx, 3], colors_codes[t_idx, 4]),
                            lw=1.4,
                            ls= :solid, #rand([:dash; :dot]),
                            label= "t = "*string(round(tₛ[ts_plot[t_idx]], digits=2)),
                            size=(900,400),
                            dpi=200,
                            legend=:outerright,
                            frame_style=:box,
                            grid=false)
                    png(plote_level_curves_Gamma,
                            figures_path*"neuron$(i)/Gamma($(N))_level_curves_ij$(i)_$(j)_no_translation"
                        )
            end


            plote_level_curves_Gamma_minusEQ= plot()

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
                plot!(tₛ[1:(ts_plot[t_idx])], Gamma[ts_plot[t_idx], 1:(ts_plot[t_idx]), i, j].-Gamma₀[ts_plot[t_idx], 1:(ts_plot[t_idx]), i, j],
                            xlabel="t′ (ms)",
                            ylabel = L"γ_{i,j}-γ_{0,ij}",
                            lc=RGBA(colors_codes[t_idx, 1], colors_codes[t_idx, 2], colors_codes[t_idx, 3], colors_codes[t_idx, 4]),
                            lw=1.4,
                            ls= :solid, #rand([:dash; :dot]),
                            label= "t = "*string(round(tₛ[ts_plot[t_idx]], digits=2)),
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

main()
