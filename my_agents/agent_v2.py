

def agent(obs):
    player = obs["player"]
    me = obs["farms"][player]
    private = obs["private"]
    fx, fy = me["farmer"]
    tile = me["tiles"][fy][fx]

    market = []

    # Buy seeds if we have no seeds and enough money
    if private["seeds"].get("WHEAT", 0) == 0 and me["money"] >= 10:
        market.append(["BUY_SEED", "WHEAT", 1])

    # Sell wheat if we have any in the shed
    wheat_in_shed = private["shed"].get("WHEAT", 0)
    if wheat_in_shed > 0:
        market.append(["SELL", "WHEAT", wheat_in_shed])

    # Plant wheat if we have seeds and the tile is empty
    if tile is None and private["seeds"].get("WHEAT", 0) > 0:
        return {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": market}

    # Water or harvest the plant if it's already planted
    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
        crop_age = obs["day"] - tile["planted_day"]
        if not tile["watered_today"]:
            return {"farmer": ["WATER"], "hands": [], "market": market}
        if crop_age >= 4:
            return {"farmer": ["HARVEST"], "hands": [], "market": market}
        

    return {"farmer": ["PASS"], "hands": [], "market": market}
