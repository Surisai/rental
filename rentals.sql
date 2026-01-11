CREATE TABLE rentals (
    rentfaster_id INT PRIMARY KEY,
    city VARCHAR(100),
    province VARCHAR(100),
    address VARCHAR(255),
    latitude DOUBLE,
    longitude DOUBLE,
    lease_term VARCHAR(50),
    type VARCHAR(50),
    price INT,
    beds INT,
    baths DOUBLE,
    sq_feet INT,
    link VARCHAR(255),
    furnishing VARCHAR(50),
    availability_date VARCHAR(50),
    smoking VARCHAR(10),
    cats VARCHAR(10),
    dogs VARCHAR(10)
);

