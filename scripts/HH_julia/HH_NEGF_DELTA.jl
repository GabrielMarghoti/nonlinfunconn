#
# Hodgkin-Huxley Neuron model NEGF
#
# Gabriel Marghoti / July 2024 Shanghai
#
using Plots
using DifferentialEquations
using LaTeXStrings
using ProgressBars
using Statistics
using Measures

gr()

function conv(X, Y, interval)
    return ((interval[end]-interval[1])/length(interval))*sum(X.*Y)
end
#=
α_m(V) = 0.1 * (V+25.0) / (exp((V+25.0)/10) - 1.0)
β_m(V) = 4.0 * exp((V) / 18.0)

α_h(V) = 0.07 * exp((V) / 20.0)
β_h(V) = 1.0 / (exp((V+30.0)/10) + 1.0)

α_n(V) = 0.01 * (V+10.0) / (exp((V+10.0)/10) - 1.0)
β_n(V) = 0.125 * exp(V / 80.0)
=#
α_m(V) = 0.1 * (V+40.0) / (1.0 - exp(-0.1 * (V+40.0)))
β_m(V) = 4.0 * exp(-(V+65.0) / 18.0)
α_h(V) = 0.07 * exp(-(V+65.0) / 20.0)
β_h(V) = 1.0 / (1.0 + exp(-0.1 * (V+35.0)))
α_n(V) = 0.01 * (V+55.0) / (1.0 - exp(-0.1 * (V+55.0)))
β_n(V) = 0.125 * exp(-(V+65.0) / 80.0)
#
# Hodgkin-Huxley model equations
function hh_model!(du, u, p, t)
    V, m, h, n = u
    C_m, g_Na_bar, g_K_bar, g_L_bar, E_Na, E_K, E_L, I_ext, stim = p

    
    if (t>=4.0 && t<=5.0) #|| (t>=10.0 && t<=11.0) || (t>=20.0 && t<=21.0)
        I_ext += stim
    end

    # Ion channel dynamics


    #=
    α_m = 0.182  * (V + 35) / (1 - exp(-(V + 35) / 9))
    β_m = -0.124 * (V + 35) / (1 - exp((V + 35) / 9))
    
    α_h = 0.25 * exp(-(V + 90) / 12)
    β_h = 0.25 * exp((V + 62) / 6) / exp((V + 90) / 12)

    α_n = 0.02   * (V - 25) / (1 - exp(-(V - 25) / 9))
    β_n = -0.002 * (V - 25) / (1 - exp((V - 25) / 9))
    =#

    du[1] = (I_ext - g_Na_bar*m^3*h*(V - E_Na) - g_K_bar*n^4*(V - E_K) - g_L_bar*(V - E_L)) / C_m
    du[2] = α_m(V) * (1.0 - m) - β_m(V) * m
    du[3] = α_h(V) * (1.0 - h) - β_h(V) * h
    du[4] = α_n(V) * (1.0 - n) - β_n(V) * n
end


function main()
        
    # Initial conditions: V, m, h, n
    V0, m0, h0, n0 = -65.0, 0.05, 0.6, 0.32
    u0 = [V0, m0, h0, n0]

    # Time span
    ti =   0.0
    tf =  30.0
    
    resolution = 1200 

    tspan = (ti, tf)

    tₛ = range(ti, tf, length=resolution)

    # Parameters

    # Define constants
    C_m      =   1.0         # membrane capacitance, in uF/cm^2
    g_Na_bar =  120.0         # maximum conductances, in mS/cm^2
    g_K_bar  =  36.0
    g_L_bar  =   0.3
    E_Na     =  50.0         # Nernst reversal potentials, in mV
    E_K      = -77.0
    E_L      = -55.0 

    # External current
    I_ext = 0.0  # in μA/cm^2
    stim  = 10.0  # in μA/cm^2

    figures_path = "/home/gabriel/figuras/qualification/NEGF_HH_DELTA/$(C_m)_$(g_Na_bar)_$(g_K_bar)_$(g_L_bar)_$(E_Na)_$(E_K)_$(E_L)_$(I_ext)_$(stim)/"
    mkpath(figures_path)

    # Solve the differential equations TRANSIENT
    prob = ODEProblem(hh_model!, u0, tspan, [C_m, g_Na_bar, g_K_bar, g_L_bar, E_Na, E_K, E_L, I_ext, 0.0])
    sol = solve(prob, Tsit5(),  dt=0.001, saveat=tₛ, reltol=1e-9, abstol=1e-9, maxiters = 1e7)

    png(plot(sol, layout=(4,1), lc=:black, xlabel=["" "" "" "Time (ms)"], ylabel=["V(t) (mV)" "m(t)" "h(t)" "n(t)"], label="", frame_style=:box, size=(500,500), dpi=200), figures_path*"HH_variables_simulation_transient")


    u0 = sol[end]
    V0, m0, h0, n0 = u0

    # Solve the differential equations STIMULATED
    prob = ODEProblem(hh_model!, u0, tspan, [C_m, g_Na_bar, g_K_bar, g_L_bar, E_Na, E_K, E_L, I_ext, stim])
    sol = solve(prob, Tsit5(),  dt=0.001, saveat=tₛ, reltol=1e-9, abstol=1e-9, maxiters = 1e7)

    var_plots = plot(sol, layout=(4,1), 
    lc=:black, xlabel=["" "" "" "Time (ms)"], ylabel=["V(t) (mV)" "m(t)" "h(t)" "n(t)"], label="", 
    frame_style=:box, size=(900,500), dpi=200, grid=false)
    hline!([V0 m0 h0 n0], label="", lc=:black, ls=:dash)
    png(var_plots,
     figures_path*"HH_variables_simulation")

    V = zeros(resolution)
    m = zeros(resolution)
    h = zeros(resolution)
    n = zeros(resolution)
    
    for t=1:resolution
        V[t] = sol[t][1]
        m[t] = sol[t][2]
        h[t] = sol[t][3]
        n[t] = sol[t][4]
    end

    n_inf = α_n.(V) ./ (α_n.(V) .+ β_n.(V))
    tau_n = 1. ./ (α_n.(V) .+ β_n.(V))
    n_inf0 = α_n(V[1]) / (α_n(V[1]) + β_n(V[1]))

    m_inf = α_m.(V) ./ (α_m.(V) .+ β_m.(V))
    tau_m = 1. ./ (α_m.(V) .+ β_m.(V))
    m_inf0 = α_m(V[1]) / (α_m(V[1]) + β_m(V[1]))

    h_inf = α_h.(V) ./ (α_h.(V) .+ β_h.(V))
    tau_h = 1. ./ (α_h.(V) .+ β_h.(V))
    h_inf0 = α_h(V[1]) / (α_h(V[1]) + β_h(V[1]))

    vi_range = range(-85,55, length=resolution)

    n_func_vi =  α_n.(vi_range) ./ (α_n.(vi_range) .+ β_n.(vi_range))
    m_func_vi =  α_m.(vi_range) ./ (α_m.(vi_range) .+ β_m.(vi_range))
    h_func_vi =  α_h.(vi_range) ./ (α_h.(vi_range) .+ β_h.(vi_range))


    png(
        plot([vi_range vi_range vi_range], [n_func_vi m_func_vi h_func_vi],
            xflip=false,
            xlabel = L"𝒗_𝒙",
            label= [L"𝒏" L"𝒎" L"𝒉"],
            ylabel = L"𝒙",
            title="Equivalent Potentials",
            size=(500,400),
            lc=[:black :red :blue],

            ls=:solid,
            dpi=200, 
            frame_style=:box,
            legend=:right,
            grid=false),
            figures_path*"equivalent_potential"
        )

    tau_n_const = mean(tau_n)

    cond_plote = plot(tₛ, [V (g_Na_bar*m.^(3).*h) (g_K_bar*n.^4)], layout=(3,1), 
    lc=:black, xlabel=["" "" L"Time\ (ms)"], ylabel=[L"V(t)\ (mV)" L"g_{Na} (t)" L"g_{K} (t)"], label="", 
    frame_style=:box, size=(900,500), dpi=200, left_margin=5mm, grid=false)
    hline!([V0 (g_Na_bar*m0.^(3).*h0) (g_K_bar*n0.^4)], label=["Equilibrium" "" "" ""], lc=:red, ls=:dash)
    png(cond_plote,
     figures_path*"HH_conductances_simulation")

     png(plot(tₛ, [(tau_n) (tau_m) (tau_h)], layout=(3,1), 
     lc=:black, xlabel=["" "" L"Time\ (ms)"], ylabel=[L"n" L"m" L"h"], label="", 
     frame_style=:box, size=(500,500), dpi=200, grid=false),
      figures_path*"HH_taus")

      png(plot(tₛ, [(n_inf) (m_inf) (h_inf)], layout=(3,1), 
      lc=:black, xlabel=["" "" L"Time\ (ms)"], ylabel=[L"n" L"m" L"h"], label="", 
      frame_style=:box, size=(500,500), dpi=200, grid=false),
       figures_path*"HH_saturation variables")
#   NONequilibrium direct green functions
    σ_n_V  = zeros(resolution, resolution)
    κ_gK_V = zeros(resolution, resolution)
    
    σ_m_V  = zeros(resolution, resolution)
    σ_h_V  = zeros(resolution, resolution)

    κ_gNa_V = zeros(resolution, resolution)
    
    conv_sigma_n_V = zeros(resolution) # n approx
    conv_sigma_m_V = zeros(resolution) # m approx
    conv_sigma_h_V = zeros(resolution) # h approx

    tau_i = (C_m)/(g_L_bar+(g_Na_bar*m0.^(3).*h0)+(g_K_bar*n0.^4))

    
    Χ_K  = zeros(resolution, resolution)
    Χ_Na = zeros(resolution, resolution)

    Χ_i  = zeros(resolution, resolution)

    for itr=ProgressBar(1:100)  #### iterative method to approximate σ

        for t=1:resolution
            for t′=1:(t) 
                nume_n = n_inf[t′]- (n_inf0 + (conv_sigma_n_V[t′]))
                nume_m = m_inf[t′]- (m_inf0 + (conv_sigma_m_V[t′]))
                nume_h = h_inf[t′]- (h_inf0 + (conv_sigma_h_V[t′]))
                if nume_n == 0.0
                    σ_n_V[t, t′]  = 0.0
                elseif abs(tau_n[t′])>=1.0e-1
                    σ_n_V[t, t′]  = nume_n / (tau_n[t′]*(V[t′].-V0))
                end
                if nume_m == 0.0
                    σ_m_V[t, t′]  = 0.0
                else
                    σ_m_V[t, t′]  = nume_m / (tau_m[t′]*(V[t′].-V0))
                end
                if nume_h == 0.0
                    σ_h_V[t, t′]  = 0.0
                else
                    σ_h_V[t, t′]  = nume_h / (tau_h[t′]*(V[t′].-V0))
                end

                κ_gK_V[t, t′] = g_K_bar*(conv_sigma_n_V[t′]^3)*((n_inf[t′]- n_inf0) - (conv_sigma_n_V[t′]))/(tau_m[t′]*(V[t′].-V0))

                κ_gNa_V[t, t′] = (g_K_bar*conv_sigma_m_V[t′]^2)*(conv_sigma_h_V[t′]*(((m_inf[t′]- m_inf0) - conv_sigma_m_V[t′])/(tau_m[t′]*(V[t′].-V0))) + conv_sigma_m_V[t′]*(((h_inf[t′]- h_inf0) - conv_sigma_h_V[t′])/(tau_h[t′]*(V[t′].-V0))))

                Χ_K[t, t′]  = -exp(-(t-t′)/tau_i)*(V[t′] - E_K )/C_m
                Χ_Na[t, t′] = -exp(-(t-t′)/tau_i)*(V[t′] - E_Na)/C_m
            end
            conv_sigma_n_V[t] = conv(σ_n_V[t, 1:t],(V[1:t].-V0), tₛ[1:t])
            conv_sigma_m_V[t] = conv(σ_m_V[t, 1:t],(V[1:t].-V0), tₛ[1:t])
            if  conv_sigma_m_V[t]>=1.0 && t>1
                conv_sigma_m_V[t] = conv_sigma_m_V[t-rand(1:(t-1))]
            end
            conv_sigma_h_V[t] = conv(σ_h_V[t, 1:t], (V[1:t].-V0), tₛ[1:t])
            for t′=1:t
                Χ_i[t,t′] = conv(Χ_K[t, t′:t], κ_gK_V[t′:t, t′], tₛ[t′:t]) + conv(Χ_Na[t, t′:t], κ_gNa_V[t′:t, t′], tₛ[t′:t])
            end
        end
            
        ts_plot = [20; 100; 150; 180; 200; 220; 250; 280; 300; 400; 1000]
        if itr%10==0 || itr==1
            png(
                heatmap(tₛ, tₛ, Χ_i,
                        xflip=false,
                        ylabel="t (current time)",
                        xlabel = "t′ (past time)",
                        size=(500,400),
                        title= "χᵢ(t,t′)",
                        fillcolor=:jet1,
                        dpi=200, 
                        frame_style=:box,
                        clims = (-1.0,1.0),
                        right_margin=2mm,
                        grid=false),
                        figures_path*"Χ_i"
                )

                plote = plot()
                
                vline!([4; 5],
                lw=1,
                la=1.0,
                label="",
                lc=[:gray],
                ls=[:solid],
                annotation=(2.5, 1.8, "Stimulus")
                )
                for t_plot in ts_plot
                    plot!(tₛ[1:(t_plot)] , Χ_i[t_plot, 1:(t_plot)],
                    ylabel= "χᵢ(t,t′)",
                    label="t=$(round(tₛ[t_plot],digits=1))",
                    xlabel = "t′ (ms)",
                    size=(900,600),
                    lc=RGBA(1-t_plot/resolution, 0, t_plot/resolution, 1),
                    ls=[:solid],
                    lw=2.5,
                    dpi=200, 
                    frame_style=:box,
                    ylims = (-1.0,1.0),
                    xlims=(0,tₛ[end]),
                    left_margin=2mm,
                    grid=false)
                end
            
                plot!(tₛ[1:(end)] , Χ_i[end, 1:(end)],
                ylabel= "χᵢ(t,t′)",
                label="t=$(round(tₛ[end],digits=1))",
                xlabel = "t′ (ms)",
                size=(1000,800),
                lw=1,
                la=1.0,
                lc=[:black],
                ls=[:solid],
                dpi=200, 
                frame_style=:box,
                #ylims = (-0.1,0.1),
                grid=false)
                    png(plote,figures_path*"level_CHI_i")



            png(
                heatmap(tₛ, tₛ, σ_n_V,
                        xflip=false,
                        ylabel="t (current time)",
                        xlabel = "t′ (past time)",
                        size=(500,400),
                        title=L"σ_{n,V}(t,t′)",
                        fillcolor=:jet1,
                        dpi=200, 
                        frame_style=:box,
                        clims = (-0.1,0.1),
                        grid=false),
                        figures_path*"sigma_n_V"
                )
                png(
                    heatmap(tₛ, tₛ, κ_gK_V,
                            xflip=false,
                            ylabel="t (current time)",
                            xlabel = "t′ (past time)",
                            size=(500,400),
                            title=L"κ_{K,V}(t,t′)",
                            fillcolor=:jet1,
                            dpi=200, 
                            frame_style=:box,
                            clims = (-0.1,0.1),
                            grid=false),
                            figures_path*"Gamma_n_V"
                    )

            png(
                heatmap(tₛ, tₛ, σ_m_V,
                        xflip=false,
                        ylabel="t (current time)",
                        xlabel = "t′ (past time)",
                        size=(500,400),
                        title=L"σ_{m,V}(t,t′)",
                        fillcolor=:jet1,
                        dpi=200, 
                        frame_style=:box,
                        clims = (-0.1,0.1),
                        grid=false),
                        figures_path*"sigma_m_V"
                )
    
            png(
                heatmap(tₛ, tₛ, σ_h_V,
                        xflip=false,
                        ylabel="t (current time)",
                        xlabel = "t′ (past time)",
                        size=(500,400),
                        title=L"σ_{h,V}(t,t′)",
                        fillcolor=:jet1,
                        dpi=200, 
                        frame_style=:box,
                        clims = (-0.1,0.1),
                        grid=false),
                        figures_path*"sigma_h_V"
                )

            png(
                plot(tₛ[end] .- tₛ, [σ_n_V[end, 1:end] ],
                        xflip=false,
                        ylabel= [L"σ_{n,V}(t,t′)"],
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
                        figures_path*"level_curve_sigma_n_V"
                )
                png(
                    plot(tₛ[end] .- tₛ, [κ_gK_V[end, 1:end] ],
                            xflip=false,
                            ylabel= [L"κ_{K,V}(t,t′)"],
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
                            figures_path*"level_curve_Gamma_K_V"
                    )
                    png(
                        plot(tₛ[end] .- tₛ, [κ_gNa_V[end, 1:end] ],
                                xflip=false,
                                ylabel= [L"κ_{Na,V}(t,t′)"],
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
                                figures_path*"level_curve_Gamma_Na_V"
                        )
                png(
                    plot(tₛ, [σ_m_V[end, 1:end] ],
                            xflip=false,
                            ylabel= [L"σ_{m,V}(t,t′)"],
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
                            figures_path*"level_curve_sigma_m_V"
                    )
                    png(
                        plot(tₛ[end] .- tₛ, [σ_m_V[end, 1:end] ],
                                xflip=false,
                                ylabel= [L"σ_{h,V}(t,t′)"],
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
                                figures_path*"level_curve_sigma_h_V"
                        )
            plot_conv_n_V = plot(tₛ, [n.-n0 conv_sigma_n_V],     
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
                figures_path*"plot_n_simulated_estimated_itr$(itr)"
            )
            plot_conv_n_V = plot(tₛ, [m.-m0 conv_sigma_m_V],     
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
                figures_path*"plot_m_simulated_estimated_itr$(itr)"
            )
            plot_conv_n_V = plot(tₛ, [h.-h0 conv_sigma_h_V],     
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
                figures_path*"plot_h_simulated_estimated_itr$(itr)"
            )
        end
    end
#


#=
####################################################################################
#### Equilibrium direct green functions
    σ₀  = zeros(resolution, resolution, N, N)
    gg₀ = zeros(resolution, resolution, N, N)
    gs₀ = zeros(resolution, resolution, N, N)
    g₀  = zeros(resolution, resolution, N, N)

    for j=1:N
        for i=j:N
            for t′=1:resolution
                σ₀[:, t′, i, j]  = Θ.(tₛ.-tₛ[t′]) .*a_r[i,j]*(1-Seq[i, j]).*dΦ(Veq[j], β[i,j], V_th[i,j]).*exp.(-(tₛ.-tₛ[t′]).*(a_d[i,j]-a_r[i,j]/(1 + exp(-β[i,j]*(Veq[j]-V_th[i,j])))))
                gg₀[:, t′, i, j] = Θ.(tₛ.-tₛ[t′]) .*κg[i, j].*exp.(-(tₛ.-tₛ[t′]).*(κ[i].+sum(κg[i, :]).+sum(κs[i, :].*Seq[i, :])))
                gs₀[:, t′, i, j] = Θ.(tₛ.-tₛ[t′]) .*κs[i, j].*(Es[i,j]-Veq[i]).*exp.(-(tₛ.-tₛ[t′]).*(κ[i].+sum(κg[i, :]).+sum(κs[i, :].*Seq[i, :])))
            end
            png(
                heatmap(tₛ, tₛ, σ₀[:, :, i, j],
                    xflip=false,
                    ylabel="t (current time)",
                    xlabel = "t′ (past time)",
                    size=(500,400),
                    fillcolor=:jet1,
                    dpi=200, 
                    frame_style=:box,
                    grid=false),
                figures_path*"sigmaZERO_ij$(i)_$(j)"
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
                figures_path*"gsZERO_ij$(i)_$(j)"
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
                figures_path*"ggZERO_ij$(i)_$(j)"
            )
        end
    end

    for t=1:resolution
        for t′=1:(t-1)
            for i=1:N
                for j=1:N
                    g₀[t, t′, i, j] = gg₀[t, t′, i, j] + conv(gs₀[t, t′:t, i, j], σ₀[t′:t, t′, i, j], tₛ[t′:t])
                end
            end
        end
    end
    
##############################################################################################
#### Integrate the system to get the ΔV and ΔS deviations from the equilibrium

    p = [N, κ, Ec, κg, κs, Es, a_r, a_d, β, V_th, I, 0.0, -1.0, C]

    ds = ODEProblem(c_elegans_model, u0, (0.0, tf), p)
    tr = solve(ds, Tsit5(), dt=0.001, saveat=tₛ, reltol=1e-9, abstol=1e-9, maxiters = 1e7)

    Vs = zeros(resolution, N)
    Ss = zeros((resolution, N, N))

    ΔVs = zeros(resolution, N)
    ΔSs = zeros((resolution, N, N))

    for t=1:resolution
        Vs[t, :] = tr[t][:, 1]
        ΔVs[t, :] = tr[t][:, 1] .- Veq
        Ss[t, :, :] = tr[t][:, 2:end]
        ΔSs[t, :, :] = tr[t][:, 2:end] .- Seq
    end

    # Plot variables trajectory
    png(
        plot(tₛ, ΔVs,
            label=["N1" "N2" "N3" "N4"],
            xlabel="time (ms)",
            ylabel = "ΔV",
            size=(500,400),
            dpi=200, 
            frame_style=:box,
            grid=false),
            figures_path*"potentials_duration_stim"
        )
#

###########################################################################
#### NONequilibrium direct green functions#
    σ = zeros(resolution, resolution, N, N)
    𝜋 = zeros(resolution, resolution, N, N)
    g = zeros(resolution, resolution, N, N)


    conv_sigma_V = zeros(resolution, N, N) # ΔS
    for itr=1:3  #### iterative method to approximate σ
        for j=1:N
            for i=j:N
                for t=1:resolution
                    for t′=1:(t-1)   
                        if Vs[t′, j] == Veq[j]
                            σ[t, t′, i, j]  = 0.0 #(σ₀[t, t′, i, j]/dΦ(Veq[j], β[i, j], V_th[i, j]))*0*(1 - (Ss[t′, i, j]-Seq[i, j])/(1-Seq[i, j]))
                        else
                            σ[t, t′, i, j]  = (σ₀[t, t′, i, j]/dΦ(Veq[j], β[i, j], V_th[i, j]))*((Φ(Vs[t′, j], β[i, j], V_th[i,j])-Φ(Veq[j], β[i, j], V_th[i,j]))/ΔVs[t′, j])*(1 - (conv_sigma_V[t′, i, j])/(1-Seq[i, j]))
                        end
                    end
                end
                for t=1:resolution
                    conv_sigma_V[t, i, j] = conv(σ[t, 1:t, i, j], ΔVs[1:t, j], tₛ[1:t])
                end
                        
            for t=1:resolution
                for t′=1:(t-1) 
                    𝜋[t, t′, i, j] = conv(gs₀[t, t′:t, i, j], (1 .-(ΔVs[t′:t, i]/(Es[i, j]-Veq[i]))).*σ[t′:t, t′, i, j], tₛ[t′:t])
                    g[t, t′, i, j] = gg₀[t, t′, i, j] + 𝜋[t, t′, i, j]
                end
            end
                    
                png(
                    heatmap(tₛ, tₛ, σ[:, :, i, j],
                            xflip=false,
                            ylabel="t (current time)",
                            xlabel = "t′ (past time)",
                            size=(500,400),
                            title="σ_$(i),$(j)(t,t′)",
                            fillcolor=:jet1,
                            dpi=200, 
                            frame_style=:box,
                            grid=false),
                            figures_path*"sigma_ij$(i)_$(j)"
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
                            figures_path*"g_ij$(i)_$(j)"
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
                            figures_path*"𝜋_ij$(i)_$(j)"
                    )
            end
        end
        plot_Delta_S_sigma_V = plot(tₛ, [ΔSs[:, 2, 1] ΔSs[:, 3, 2] ΔSs[:, 4, 1]  ΔSs[:, 4, 3]],     # plot only the synapses that matter for the networks (4←3←2←1)
        label=["S [2←1]" "S [3⇐2]" "S [4⥒1]" "S [4←3]"],
        size=(500,400),
        lc = [:gray :black :red :blue],
        xlabel="time (ms)",
        ylabel = "ΔS",
        dpi=200, 
        frame_style=:box,
        grid=false)
        plot!(tₛ, [conv_sigma_V[:, 2, 1] conv_sigma_V[:, 3, 2] conv_sigma_V[:, 4, 1]  conv_sigma_V[:, 4, 3]],     # plot only the synapses that matter for the networks (4←3←2←1)
        label="",
        lc = [:gray :black :red :blue],
        ls=:dot,
        lw=2,
        size=(500,400),
        xlabel="time (ms)",
        ylabel = "ΔS",
        dpi=200, 
        frame_style=:box,
        grid=false),
        png(plot_Delta_S_sigma_V,       
            figures_path*"synapses_duration_stim_itr$(itr)"
        )
    end


        plot_Delta_S_sigma_V = plot(tₛ, [ΔSs[:, 2, 1] ΔSs[:, 3, 2] ΔSs[:, 4, 1]  ΔSs[:, 4, 1]],     # plot only the synapses that matter for the networks (4←3←2←1)
        label=["S [2←1]" "S [3⇐2]" "S [4⥒1]" "S [4←3]"],
        lc = [:gray :black :red :blue],
        size=(500,400),
        xlabel="time (ms)",
        ylabel = "ΔS",
        dpi=200, 
        frame_style=:box,
        grid=false)
        plot!(tₛ, [conv_sigma_V[:, 2, 1] conv_sigma_V[:, 3, 2] conv_sigma_V[:, 4, 1]  conv_sigma_V[:, 4, 3]],     # plot only the synapses that matter for the networks (4←3←2←1)
        label="",
        lc = [:gray :black :red :blue],
        ls=:dot,
        lw=2,
        size=(500,400),
        xlabel="time (ms)",
        ylabel = "ΔS",
        dpi=200, 
        frame_style=:box,
        grid=false),
    png(plot_Delta_S_sigma_V,       
        figures_path*"synapses_duration_stim"
    )
   
#######################################################################################
##### CONNECTED Green functions
#####    
    # The connected Green function is the directed plus the convolution with the paths from node j to node i
    # First compute the trivial ones, the first neighbors of the boundary condition ΔVⱼ, then propagate to higher neighbors 
    G₀ = g₀                # First neighbors (direct)       
    
    conv_G_V = zeros(resolution, N) # test G function
    for t=1:resolution
        for t′=1:(t-1)         # Second neighbors
            G₀[t, t′, 3, 1] += conv(g₀[t, t′:t, 3, 2], G₀[t′:t, t′, 2, 1], tₛ[t′:t])
            G₀[t, t′, 4, 2] += conv(g₀[t, t′:t, 4, 3], G₀[t′:t, t′, 3, 2], tₛ[t′:t])
        end
    end
    for t=1:resolution    # Third neighbors
        for t′=1:(t-1)
            G₀[t, t′, 4, 1] += conv(g₀[t, t′:t, 4, 3], G₀[t′:t, t′, 3, 1], tₛ[t′:t])
        end
    end

    G = g                # First neighbors (direct)                        
    for t=1:resolution
        for t′=1:(t-1)         # Second neighbors
            G[t, t′, 3, 1] += conv(g[t, t′:t, 3, 2], G[t′:t, t′, 2, 1], tₛ[t′:t])
            G[t, t′, 4, 2] += conv(g[t, t′:t, 4, 3], G[t′:t, t′, 3, 2], tₛ[t′:t])
        end
    end
    for t=1:resolution    # Third neighbors
        for t′=1:(t-1)
            G[t, t′, 4, 1] += conv(g[t, t′:t, 4, 3], G[t′:t, t′, 3, 1], tₛ[t′:t])
        end
    end
    for i=1:N
        for t=1:resolution
            for j=1:N
                conv_G_V[t, i] += conv(g[t, 1:t, i, j], ΔVs[1:t, j], tₛ[1:t])
            end
        end
    end


    for j=1:N-1
        for i=j:N
            png(
                heatmap(tₛ, tₛ, G₀[:, :, i, j],
                    xflip=false,
                    ylabel="t (current time)",
                    xlabel = "t′ (past time)",
                    size=(500,400),
                    title="G₀$(i),$(j)(t,t′)",
                    fillcolor=:jet1,
                    dpi=200, 
                    frame_style=:box,
                    grid=false),
                    figures_path*"G0_ij$(i)_$(j)"
                )
            png(
                heatmap(tₛ, tₛ, G[:, :, i, j],
                    xflip=false,
                    ylabel="t (current time)",
                    xlabel = "t′ (past time)",
                    title="G$(i),$(j)(t,t′)",
                    size=(500,400),
                    fillcolor=:jet1,
                    dpi=200, 
                    frame_style=:box,
                    grid=false),
                    figures_path*"G_ij$(i)_$(j)"
                )

            plote_level_curves_G = plot(tₛ[end].-tₛ[1:end], G₀[end, 1:end, i, j],
                            xlabel="t-t′",
                            ylabel = "G",
                            label= "G₀",
                            lc=:black,
                            lw=2.4,
                            size=(500,400),
                            dpi=200, 
                            frame_style=:box,
                            grid=false)
            for t_idx in probe_times_idx
                plot!(tₛ[t_idx].-tₛ[2:(t_idx)], G[t_idx, 2:(t_idx), i, j],
                            xlabel="t-t′",
                            ylabel = "G",
                            lc = RGBA(t_idx/plot_times_idx[end], 0, 1 - (2*t_idx/plot_times_idx[end] -1)^2, 1),
                            la=0.9,
                            lw=1.4,
                            ls= :solid, #rand([:dash; :dot]),
                            label= "t = "*string(round(tₛ[t_idx], digits=2)),
                            size=(500,400),
                            dpi=200, 
                            frame_style=:box,
                            grid=false)
                    png(plote_level_curves_G,
                            figures_path*"G_level_curves_ij$(i)_$(j)"
                        )
            end
        end
    end

      # Plot variables trajectory
      plot_DV_GDV = plot(tₛ, ΔVs,
            label=["N1" "N2" "N3" "N4"],
            xlabel="time (ms)",
            lc = [:gray :black :red :blue],
            ylabel = "ΔV",
            size=(500,400),
            dpi=200, 
            frame_style=:box,
            grid=false)
        plot!(tₛ, conv_G_V,
            label="",
            xlabel="time (ms)",
            ls=:dot,
            lc = [:gray :black :red :blue],
            ylabel = "ΔV",
            size=(500,400),
            dpi=200, 
            frame_style=:box,
            grid=false)
      png(
        plot_DV_GDV,
            figures_path*"potentials_duration_stim"
        )

#################################################################################################
####### Response Synaptic Susceptibility

    χ     = zeros(resolution, resolution, N, N)  

    conv_sigma0_V_chi = zeros(resolution, resolution, N, N)

    for itr=1:3
        for j=1:N
            for i=j:N
                for t=1:resolution
                    for t′=1:(t-1)    
                        χ[t, t′, i, j]  = ((σ₀[t, t′, i, j]*dΦ(Vs[t′, j], β[i, j], V_th[i, j]))/dΦ(Veq[j], β[i, j], V_th[i, j]))*(1 - (conv_sigma_V[t′, i, j])/(1-Seq[i, j])) - conv_sigma0_V_chi[t, t′, i, j]*(1/(dΦ(Veq[j], β[i, j], V_th[i, j])*(1-Seq[i,j])))   
                    end
                end
                for t=1:resolution
                    for t′=1:(t-1)    
                        conv_sigma0_V_chi[t, t′, i, j] = conv(σ₀[t, t′:t, i, j].*(Φ.(Vs[t′:t, j], β[i, j], V_th[i,j]).-Φ(Veq[j], β[i, j], V_th[i,j])), χ[t′:t, t′, i, j], tₛ[t′:t])
                    end
                end

            png(
                heatmap(tₛ, tₛ, χ[:, :, i, j],
                            xflip=false,
                            ylabel="t (current time)",
                            xlabel = "t′ (past time)",
                            size=(500,400),
                            title="χ_$(i),$(j)(t,t′)",
                            fillcolor=:jet1,
                            dpi=200, 
                            frame_style=:box,
                            grid=false),
                            figures_path*"Chi_ij$(i)_$(j)_itr$(itr)"
                    )
            end
        end
    end

################################################################
#### Direct response function

    f     = zeros(resolution, resolution, N, N)

    for itr=1:3
        for j=1:N
            for i=j:N
                for t=1:resolution
                    for t′=1:(t-1)
                        f[t, t′, i, j] = gg₀[t, t′, i, j]  + conv(gs₀[t, t′:t, i, j], (1 .-(ΔVs[t′:t, i]./(Es[i, j]-Veq[i]))).*χ[t′:t, t′, i, j] .- conv_sigma_V[t′:t, i, j].*f[t′:t, t′, i, j], tₛ[t′:t])
                    end
                end

                png(
                    heatmap(tₛ, tₛ, f[:, :, i, j],
                            xflip=false,
                            ylabel="t (current time)",
                            xlabel = "t′ (past time)",
                            title="f_$(i),$(j)(t,t′)",
                            size=(500,400),
                            fillcolor=:jet1,
                            dpi=200, 
                            frame_style=:box,
                            grid=false),
                            figures_path*"f_ij$(i)_$(j)_itr$(itr)"
                    )
            end
        end
    end
                
##################################################################################################
#### Connected response functions

    F = f                # First neighbors (direct)         
                   
    for t=1:resolution
        for t′=1:(t-1)         # Second neighbors
            F[t, t′, 3, 1] += conv(f[t, t′:t, 3, 2], f[t′:t, t′, 2, 1], tₛ[t′:t])
            F[t, t′, 4, 2] += conv(f[t, t′:t, 4, 3], f[t′:t, t′, 3, 2], tₛ[t′:t])
        end
    end
    for t=1:resolution    # Third neighbors
        for t′=1:(t-1)
            F[t, t′, 4, 1] += conv(f[t, t′:t, 4, 3], F[t′:t, t′, 3, 1], tₛ[t′:t])
        end
    end

    for j=1:N
        for i=j:N

        png(
            heatmap(tₛ, tₛ, F[:, :, i, j],
                xflip=false,
                ylabel="t (current time)",
                xlabel = "t′ (past time)",
                title="F$(i),$(j)(t,t′)",
                size=(500,400),
                fillcolor=:jet1,
                dpi=200, 
                frame_style=:box,
                grid=false),
                figures_path*"F_ij$(i)_$(j)"
            )
        end
    end
    
    plote_level_curves_F = plot(tₛ[end].-tₛ[1:end], G₀[end, 1:end, 4, 1],
            xlabel="t-t′",
            ylabel = "F",
            label= "F₀",
            lc=:black,
            lw=2.4,
            size=(500,400),
            dpi=200, 
            frame_style=:box,
            grid=false)
    for t_idx in plot_times_idx
        plot!(tₛ[t_idx].-tₛ[2:(t_idx)], F[t_idx, 2:(t_idx), 4, 1],
                        xlabel="t-t′",
                        ylabel = "F",
                        la=0.9,
                        lw=1.4,
                        lc = RGBA(t_idx/plot_times_idx[end], 0, 1 - (2*t_idx/plot_times_idx[end] -1)^2, 1),
                        ls=:solid, #rand([:dash; :dot]),
                        label= "t = "*string(round(tₛ[t_idx], digits=2)),
                        size=(500,400),
                        dpi=200, 
                        frame_style=:box,
                        grid=false)
    end
    png(plote_level_curves_F,
        figures_path*"F_level_curves_4_1"
        )
    


    # Simulate the system stimulated by strong current I and small probes i
#=
    𝛿Vs = zeros(length(probe_times_idx), resolution, N)
    𝛿Ss = zeros(length(probe_times_idx), resolution, N, N)

    for idx=1:length(probe_times_idx)
        ds = ODEProblem(c_elegans_model, u0, (0.0, tf), [N, κ, Ec, κg, κs, Es, a_r, a_d, β, V_th, I, i, tₛ[probe_times_idx[idx]], C])
        tr = solve(ds, Tsit5(), dt=0.001, saveat=tₛ, reltol=1e-9, abstol=1e-9, maxiters = 1e7)
    
        for t=1:resolution
            𝛿Vs[idx, t, :] = tr[t][:, 1] .- Vs[t, :]
            𝛿Ss[idx, t, :, :] = tr[t][:, 2:end] .- Ss[t, :, :]
        end
    end
=#
=#

end
main()
