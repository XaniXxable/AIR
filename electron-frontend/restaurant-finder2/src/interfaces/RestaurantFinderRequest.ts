export interface RestaurantFinderRequest {
    Filters: Filter[];
    UserInput: string;
}

export interface Filter {
    FilterName: string;
    FilterValue: boolean;
}